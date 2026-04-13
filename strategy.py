#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume regime filter
    # Long: close > H4 (Camarilla resistance) AND 1w volume > 1.5x 20-week avg
    # Short: close < L4 (Camarilla support) AND 1w volume > 1.5x 20-week avg
    # Exit: close < H3 for longs OR close > L3 for shorts
    # Using discrete sizing 0.25 to minimize fee churn.
    # Camarilla levels provide institutional price levels effective in both bull/bear markets.
    # Weekly volume filter ensures trades occur during institutional participation.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(20, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-20:i])
    
    # Volume spike condition: current weekly volume > 1.5x 20-week average
    vol_spike_1w = np.zeros(len(vol_1w), dtype=bool)
    for i in range(20, len(vol_1w)):
        if not np.isnan(vol_ma_1w[i]):
            vol_spike_1w[i] = vol_1w[i] > (1.5 * vol_ma_1w[i])
    
    # Align weekly volume spike to daily
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Need prior day's high/low/close for Camarilla calculation
        if i == 0:
            continue
            
        # Calculate Camarilla levels from previous day
        ph = high[i-1]  # previous high
        pl = low[i-1]   # previous low
        pc = close[i-1] # previous close
        
        # Camarilla levels
        range_ = ph - pl
        if range_ <= 0:
            signals[i] = 0.0
            continue
            
        # Resistance levels
        h4 = pc + (range_ * 1.1/2)
        h3 = pc + (range_ * 1.1/4)
        h6 = pc + (range_ * 1.1/2) * 1.166  # approx H6
        
        # Support levels
        l4 = pc - (range_ * 1.1/2)
        l3 = pc - (range_ * 1.1/4)
        l6 = pc - (range_ * 1.1/2) * 1.166  # approx L6
        
        # Volume confirmation from weekly data
        vol_confirm = vol_spike_1w_aligned[i] > 0.5  # boolean as float
        
        # Entry conditions
        long_entry = (close[i] > h4) and vol_confirm
        short_entry = (close[i] < l4) and vol_confirm
        
        # Exit conditions
        long_exit = (position == 1) and (close[i] < h3)
        short_exit = (position == -1) and (close[i] > l3)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_exit:
            position = 0
            signals[i] = 0.0
        elif short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0