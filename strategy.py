#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ThreeLineBreak_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data (same timeframe for Three Line Break)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1d: Three Line Break calculation ===
    close_1d = df_1d['close'].values
    
    # Initialize Three Line Break arrays
    tlb_up = np.full(len(close_1d), np.nan)
    tlb_down = np.full(len(close_1d), np.nan)
    
    # First bar: no reversal line yet
    if len(close_1d) > 0:
        tlb_up[0] = close_1d[0]
        tlb_down[0] = close_1d[0]
    
    # Calculate Three Line Break
    for i in range(1, len(close_1d)):
        if i >= 3:
            # Need at least 3 prior closes to check reversal
            max_close_3 = np.max(close_1d[i-3:i])
            min_close_3 = np.min(close_1d[i-3:i])
            
            # Bullish reversal: close above last 3 highs
            if close_1d[i] > max_close_3:
                tlb_up[i] = close_1d[i]
                tlb_down[i] = tlb_down[i-1]  # carry down
            # Bearish reversal: close below last 3 lows
            elif close_1d[i] < min_close_3:
                tlb_down[i] = close_1d[i]
                tlb_up[i] = tlb_up[i-1]  # carry up
            else:
                # Continuation: carry previous values
                tlb_up[i] = tlb_up[i-1]
                tlb_down[i] = tlb_down[i-1]
        else:
            # Not enough history yet: carry previous
            tlb_up[i] = tlb_up[i-1] if i > 0 else close_1d[0]
            tlb_down[i] = tlb_down[i-1] if i > 0 else close_1d[0]
    
    # Align Three Line Break levels
    tlb_up_aligned = align_htf_to_ltf(prices, df_1d, tlb_up)
    tlb_down_aligned = align_htf_to_ltf(prices, df_1d, tlb_down)
    
    # === 1w: EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1d: ATR(14) for stop loss ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d: Volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        tlb_up_val = tlb_up_aligned[i]
        tlb_down_val = tlb_down_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_avg_val = vol_avg_20_aligned[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(tlb_up_val) or np.isnan(tlb_down_val) or 
            np.isnan(ema34_val) or np.isnan(atr_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.3x 20-period average
        vol_condition = current_volume > 1.3 * vol_avg_val
        
        if position == 0:
            # Long conditions: Three Line Break bullish reversal + above weekly EMA34 + volume
            if (current_close > tlb_up_val and 
                current_close > ema34_val and 
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions: Three Line Break bearish reversal + below weekly EMA34 + volume
            elif (current_close < tlb_down_val and 
                  current_close < ema34_val and 
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: Three Line Break bearish reversal OR stop loss
            if (current_close < tlb_down_val or 
                current_close < entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Three Line Break bullish reversal OR stop loss
            if (current_close > tlb_up_val or 
                current_close > entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals