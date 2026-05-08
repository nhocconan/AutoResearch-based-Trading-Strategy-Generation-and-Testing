#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_SuperTrend_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly SuperTrend calculation (ATR=10, multiplier=3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic SuperTrend calculation
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (3 * atr_1w)
    lower_band_1w = hl2_1w - (3 * atr_1w)
    
    # Initialize SuperTrend
    supertrend_1w = np.zeros_like(close_1w)
    uptrend_1w = np.ones_like(close_1w, dtype=bool)
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band_1w[i-1]:
            uptrend_1w[i] = True
        elif close_1w[i] < lower_band_1w[i-1]:
            uptrend_1w[i] = False
        else:
            uptrend_1w[i] = uptrend_1w[i-1]
            if uptrend_1w[i] and lower_band_1w[i] < lower_band_1w[i-1]:
                lower_band_1w[i] = lower_band_1w[i-1]
            if not uptrend_1w[i] and upper_band_1w[i] > upper_band_1w[i-1]:
                upper_band_1w[i] = upper_band_1w[i-1]
        
        if uptrend_1w[i]:
            supertrend_1w[i] = lower_band_1w[i]
        else:
            supertrend_1w[i] = upper_band_1w[i]
    
    # Trend signal: 1 for uptrend, 0 for downtrend
    trend_1w = uptrend_1w.astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Daily SuperTrend for entry/exit (ATR=10, multiplier=3)
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_d = pd.Series(tr_d).rolling(window=10, min_periods=10).mean().values
    
    hl2_d = (high + low) / 2
    upper_band_d = hl2_d + (3 * atr_d)
    lower_band_d = hl2_d - (3 * atr_d)
    
    supertrend_d = np.zeros_like(close)
    uptrend_d = np.ones_like(close, dtype=bool)
    
    for i in range(1, len(close)):
        if close[i] > upper_band_d[i-1]:
            uptrend_d[i] = True
        elif close[i] < lower_band_d[i-1]:
            uptrend_d[i] = False
        else:
            uptrend_d[i] = uptrend_d[i-1]
            if uptrend_d[i] and lower_band_d[i] < lower_band_d[i-1]:
                lower_band_d[i] = lower_band_d[i-1]
            if not uptrend_d[i] and upper_band_d[i] > upper_band_d[i-1]:
                upper_band_d[i] = upper_band_d[i-1]
        
        supertrend_d[i] = lower_band_d[i] if uptrend_d[i] else upper_band_d[i]
    
    # Daily trend for reference
    trend_d = uptrend_d.astype(float)
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and SuperTrend
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(supertrend_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above SuperTrend with volume spike and weekly uptrend
            long_cond = (close[i] > supertrend_d[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price below SuperTrend with volume spike and weekly downtrend
            short_cond = (close[i] < supertrend_d[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below SuperTrend
            if close[i] < supertrend_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above SuperTrend
            if close[i] > supertrend_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly SuperTrend trend filter with daily SuperTrend entries on volume spikes.
# Works in bull markets (trend following) and bear markets (avoids counter-trend trades).
# Weekly SuperTrend ensures alignment with longer-term trend, reducing whipsaws.
# Volume spikes confirm institutional interest in breakouts.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.