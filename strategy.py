#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
    # Long when price breaks above Kumo cloud (Senkou Span A > Senkou Span B) and TK cross bullish
    # and 1d EMA50 > EMA200 (uptrend) and volume > 1.5x average.
    # Short when price breaks below Kumo cloud and TK cross bearish
    # and 1d EMA50 < EMA200 (downtrend) and volume > 1.5x average.
    # Exit when price re-enters the cloud or TK cross reverses.
    # Uses Ichimoku as a comprehensive trend/momentum system filtered by higher timeframe trend.
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods) on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 and EMA200 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average (20-period) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready (need enough for Ichimoku calculation)
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        # Price above cloud: close > Senkou Span A and close > Senkou Span B
        price_above_cloud = (close[i] > senkou_span_a[i]) and (close[i] > senkou_span_b[i])
        # Price below cloud: close < Senkou Span A and close < Senkou Span B
        price_below_cloud = (close[i] < senkou_span_a[i]) and (close[i] < senkou_span_b[i])
        # TK cross bullish: Tenkan-sen > Kijun-sen
        tk_bullish = tenkan_sen[i] > kijun_sen[i]
        # TK cross bearish: Tenkan-sen < Kijun-sen
        tk_bearish = tenkan_sen[i] < kijun_sen[i]
        
        # Trend filter: 1d EMA50 > EMA200 for uptrend, < for downtrend
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Entry conditions
        long_signal = price_above_cloud and tk_bullish and uptrend and volume_confirm
        short_signal = price_below_cloud and tk_bearish and downtrend and volume_confirm
        
        # Exit conditions: price re-enters cloud or TK cross reverses
        long_exit = not price_above_cloud or not tk_bullish
        short_exit = not price_below_cloud or not tk_bearish
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ichimoku_trend_volume_v1"
timeframe = "6h"
leverage = 1.0