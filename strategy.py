#!/usr/bin/env python3
# exp_6459_6h_donchian20_12h_pivot_vol_v1
# Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot direction filter and volume confirmation.
# In bull markets, breakouts above R4 continue; in bear markets, breakdowns below S4 continue.
# Weekly pivot adds structural bias. Volume confirms momentum. Designed for low frequency (~20-50/year).
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Includes ATR-based stoploss.

name = "exp_6459_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    # Camarilla levels
    r4_12h = pp_12h + ((high_12h - low_12h) * 1.1 / 2.0)
    s4_12h = pp_12h - ((high_12h - low_12h) * 1.1 / 2.0)
    r3_12h = pp_12h + ((high_12h - low_12h) * 1.1 / 4.0)
    s3_12h = pp_12h - ((high_12h - low_12h) * 1.1 / 4.0)
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Donchian upper/lower (20-period lookback)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high_6h[i-20:i])
        donch_low[i] = np.min(low_6h[i-20:i])
    
    # Volume confirmation: 6h volume > 1.5 * 20-period average
    volume_6h = prices['volume'].values
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume_6h[i-20:i])
    vol_ratio = np.where(vol_ma > 0, volume_6h / vol_ma, 0)
    
    # ATR for stoploss (14-period)
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])  # Simple ATR
    
    # Initialize signals
    signals = np.zeros(n)
    
    # Track position state for stoploss
    position_side = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index 20 (enough for Donchian and volume MA)
    start_idx = max(20, 14)  # ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if any required value is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]):
            continue
            
        current_price = close_6h[i]
        
        # Check stoploss for existing position
        if position_side == 1:  # Long position
            if current_price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
        elif position_side == -1:  # Short position
            if current_price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry conditions
        long_signal = False
        short_signal = False
        
        # Long: price breaks above Donchian high AND above 12h R4 AND volume confirmation
        if (current_price > donch_high[i] and 
            current_price > r4_12h_aligned[i] and 
            vol_ratio[i] > 1.5):
            long_signal = True
            
        # Short: price breaks below Donchian low AND below 12h S4 AND volume confirmation
        if (current_price < donch_low[i] and 
            current_price < s4_12h_aligned[i] and 
            vol_ratio[i] > 1.5):
            short_signal = True
        
        # Generate signals
        if long_signal and position_side <= 0:
            signals[i] = 0.25  # Long 25%
            position_side = 1
            entry_price = current_price
        elif short_signal and position_side >= 0:
            signals[i] = -0.25  # Short 25%
            position_side = -1
            entry_price = current_price
        elif position_side == 0:
            signals[i] = 0.0  # Flat
        # If already in position and no stoploss/exit signal, hold position (signal remains unchanged)
    
    return signals