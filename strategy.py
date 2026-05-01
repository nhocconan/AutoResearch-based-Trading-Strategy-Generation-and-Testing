#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Kumo cloud AND Tenkan > Kijun (bullish momentum) AND 12h close > EMA50 (bullish trend) AND volume > 1.5x 20-bar average.
# Short when price breaks below Kumo cloud AND Tenkan < Kijun (bearish momentum) AND 12h close < EMA50 (bearish trend) AND volume > 1.5x 20-bar average.
# Ichimoku provides dynamic support/resistance via cloud, Tenkan/Kijun cross confirms momentum, 12h EMA50 filters higher timeframe trend.
# Primary timeframe: 6h, HTF: 12h for EMA trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with signal size 0.25.

name = "6h_Ichimoku_CloudBreakout_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # The Kumo (cloud) is between Senkou Span A and Senkou Span B
    # Upper cloud boundary: max(Senkou A, Senkou B)
    # Lower cloud boundary: min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe, no alignment needed but using helper for consistency
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    upper_cloud_aligned = align_htf_to_ltf(prices, prices, upper_cloud)
    lower_cloud_aligned = align_htf_to_ltf(prices, prices, lower_cloud)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Ichimoku (52 periods) and EMA
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(upper_cloud_aligned[i]) or np.isnan(lower_cloud_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Ichimoku signals
        price_above_cloud = curr_low > upper_cloud_aligned[i]  # price breaks above cloud
        price_below_cloud = curr_high < lower_cloud_aligned[i]  # price breaks below cloud
        
        bullish_momentum = tenkan_aligned[i] > kijun_aligned[i]  # Tenkan > Kijun
        bearish_momentum = tenkan_aligned[i] < kijun_aligned[i]  # Tenkan < Kijun
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above cloud AND bullish momentum AND bullish trend AND volume confirmation
            if (price_above_cloud and 
                bullish_momentum and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND bearish momentum AND bearish trend AND volume confirmation
            elif (price_below_cloud and 
                  bearish_momentum and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below cloud OR momentum turns bearish OR trend turns bearish
            if (price_below_cloud or 
                not bullish_momentum or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above cloud OR momentum turns bullish OR trend turns bullish
            if (price_above_cloud or 
                not bearish_momentum or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals