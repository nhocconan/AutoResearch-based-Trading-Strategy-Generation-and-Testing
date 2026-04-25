#!/usr/bin/env python3
"""
1d_Weekly_Keltner_Breakout_1wTrend_Filter_v1
Hypothesis: Trade weekly Keltner Channel breakouts on 1d timeframe with 1w trend filter. In bullish 1w trend (price above weekly EMA20), go long when price breaks above weekly upper Keltner (EMA20 + 2*ATR). In bearish 1w trend (price below weekly EMA20), go short when price breaks below weekly lower Keltner (EMA20 - 2*ATR). Volume confirmation (1.5x 20-bar avg) filters weak breakouts. Uses discrete position sizing (0.25) to minimize fee drag and target ~10-20 trades/year. Designed to work in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend and Keltner calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 and ATR(14) for Keltner Channel
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA20
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR(14)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first period TR
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly Keltner Channel
    keltner_upper = ema_20 + (2.0 * atr_14)
    keltner_lower = ema_20 - (2.0 * atr_14)
    
    # Align to 1d timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA20 and ATR(14)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(keltner_upper_aligned[i]) or
            np.isnan(keltner_lower_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend: price above/below weekly EMA20
        weekly_bullish = close[i] > ema_20_aligned[i]
        weekly_bearish = close[i] < ema_20_aligned[i]
        
        # Keltner breakout signals
        breakout_up = close[i] > keltner_upper_aligned[i]
        breakout_down = close[i] < keltner_lower_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            long_signal = breakout_up and volume_spike[i] and weekly_bullish
            short_signal = breakout_down and volume_spike[i] and weekly_bearish
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price crosses below weekly EMA20 (trend change)
            if close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price crosses above weekly EMA20 (trend change)
            if close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Keltner_Breakout_1wTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0