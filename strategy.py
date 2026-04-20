#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ThreeLineBreak_TrendFollowing_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Three Line Break from Daily Data ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize Three Line Break arrays
    tlb_up = np.full_like(close_1d, np.nan)
    tlb_down = np.full_like(close_1d, np.nan)
    
    # First values
    tlb_up[0] = close_1d[0]
    tlb_down[0] = close_1d[0]
    
    # Track reversal count
    reversal_count = 0
    last_close = close_1d[0]
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > tlb_up[i-1]:
            # New up line
            tlb_up[i] = close_1d[i]
            tlb_down[i] = tlb_down[i-1]
            reversal_count = 0
        elif close_1d[i] < tlb_down[i-1]:
            # New down line
            tlb_down[i] = close_1d[i]
            tlb_up[i] = tlb_up[i-1]
            reversal_count = 0
        else:
            # Inside bar - no new line
            tlb_up[i] = tlb_up[i-1]
            tlb_down[i] = tlb_down[i-1]
            reversal_count += 1
            
            # Reverse if 3 consecutive inside bars
            if reversal_count >= 3:
                if close_1d[i] > tlb_up[i-1]:
                    tlb_up[i] = close_1d[i]
                    tlb_down[i] = tlb_down[i-1]
                elif close_1d[i] < tlb_down[i-1]:
                    tlb_down[i] = close_1d[i]
                    tlb_up[i] = tlb_up[i-1]
                reversal_count = 0
    
    # Align to 6h timeframe
    tlb_up_aligned = align_htf_to_ltf(prices, df_1d, tlb_up)
    tlb_down_aligned = align_htf_to_ltf(prices, df_1d, tlb_down)
    
    # === 6h Trend Filter: EMA20 > EMA50 for uptrend ===
    close_series = pd.Series(prices['close'].values)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema20_val = ema20[i]
        ema50_val = ema50[i]
        tlb_up_val = tlb_up_aligned[i]
        tlb_down_val = tlb_down_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(ema50_val) or 
            np.isnan(tlb_up_val) or np.isnan(tlb_down_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above TLB up line with uptrend and volume
            if close_val > tlb_up_val and ema20_val > ema50_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Close below TLB down line with downtrend and volume
            elif close_val < tlb_down_val and ema20_val < ema50_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below TLB down line OR trend breaks down
            if close_val < tlb_down_val or ema20_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above TLB up line OR trend breaks up
            if close_val > tlb_up_val or ema20_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals