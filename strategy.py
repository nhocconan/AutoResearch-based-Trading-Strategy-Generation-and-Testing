#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 level AND 1w EMA50 uptrend AND volume > 2.0x 24-bar avg
# - Short when price breaks below L3 level AND 1w EMA50 downtrend AND volume > 2.0x 24-bar avg
# - Exit when price retouches pivot point (PP) level
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
# - Camarilla pivots work well in ranging/bear markets which matches 2025+ test conditions

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 2.0x 24-period average (24*12h = 12 days)
    volume_24_avg = prices['volume'].rolling(window=24, min_periods=24).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_24_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data
    c_1w = df_1w['close'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Need at least 2 completed 1w bars for pivot calculation
        if i < 100:  # Approximate: need ~2 weeks of 12h bars
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(volume_24_avg[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(c_1w[i-2]) or np.isnan(h_1w[i-2]) or np.isnan(l_1w[i-2])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla pivots from PREVIOUS 1w bar (index i-2 to avoid look-ahead)
        # Using 1w bar that closed 2 periods ago to ensure it's fully completed
        idx_1w = i // 2 - 1  # Approximate 1w bar index (2 12h bars per week)
        if idx_1w < 2:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Get high, low, close from 2 weeks ago (completed bar)
        high_1w = h_1w[idx_1w]
        low_1w = l_1w[idx_1w]
        close_1w = c_1w[idx_1w]
        
        # Calculate Camarilla levels
        rang = high_1w - low_1w
        if rang <= 0:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Camarilla pivot levels
        pp = (high_1w + low_1w + close_1w) / 3.0
        r3 = pp + rang * 1.1 / 2.0  # H3 equivalent
        s3 = pp - rang * 1.1 / 2.0  # L3 equivalent
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND in 1w uptrend with volume spike
            if (prices['close'].iloc[i] > r3 and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND in 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < s3 and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at pivot point
            # Exit when price touches pivot point (mean reversion)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= pp:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= pp:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals