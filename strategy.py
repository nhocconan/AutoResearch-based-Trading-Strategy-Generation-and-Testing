#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 1d HTF for EMA50 to capture daily trend and reduce false breakouts in choppy markets.
# Camarilla H4/L4 from 12h provides proven intraday reversal/continuation levels with good historical performance.
# Volume confirmation at 2.5x average ensures strong participation while limiting trades (~12-37/year target).
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "12h_Camarilla_H4_L4_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels H4 and L4 from 12h timeframe (using prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h bar's high, low, close for Camarilla calculation
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    # Camarilla H4 and L4 levels (proven breakout/continuation levels)
    camarilla_h4_12h = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 2
    camarilla_l4_12h = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (they are already 12h, but align for safety)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4_12h)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4_12h)
    
    # 1d EMA50 for trend filter (daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.5x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H4 AND price > 1d EMA50 AND volume spike
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 AND price < 1d EMA50 AND volume spike
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L4 OR price < 1d EMA50
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above H4 OR price > 1d EMA50
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals