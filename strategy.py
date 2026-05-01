#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Uses Tenkan-sen (9) and Kijun-sen (26) for TK cross, Senkou Span A/B for cloud.
# Long when TK cross bullish AND price above cloud AND 1d close > EMA50 (bullish trend) AND volume > 1.5x 20-bar average.
# Short when TK cross bearish AND price below cloud AND 1d close < EMA50 (bearish trend) AND volume > 1.5x 20-bar average.
# Ichimoku provides multi-line trend/momentum/cloud structure proven effective on 6h timeframe.
# Primary timeframe: 6h, HTF: 1d for EMA trend filter.

name = "6h_Ichimoku_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (using values from 26 periods ago for current price)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # 1d EMA50 trend filter
    prev_close = df_1d['close'].values
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # warmup for Ichimoku (needs 52-period data)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        
        # TK cross: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
        tk_bullish = curr_tenkan > curr_kijun
        tk_bearish = curr_tenkan < curr_kijun
        
        # Cloud filter: price above cloud (bullish) or below cloud (bearish)
        cloud_top = max(senkou_a_lagged[i], senkou_b_lagged[i])
        cloud_bottom = min(senkou_a_lagged[i], senkou_b_lagged[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: TK cross bullish AND price above cloud AND bullish trend AND volume confirmation
            if (tk_bullish and 
                price_above_cloud and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud AND bearish trend AND volume confirmation
            elif (tk_bearish and 
                  price_below_cloud and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: TK cross turns bearish OR price falls below cloud OR trend turns bearish
            if (not tk_bullish or 
                not price_above_cloud or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: TK cross turns bullish OR price rises above cloud OR trend turns bullish
            if (not tk_bearish or 
                not price_below_cloud or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals