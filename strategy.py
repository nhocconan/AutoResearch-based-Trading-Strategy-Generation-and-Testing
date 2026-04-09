#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR trailing stop
# - Uses 1d Camarilla pivot levels (H4/L4) for breakout signals
# - Confirms with 4h volume > 1.8x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivots work in ranging markets (mean reversion at S3/R3) and trending markets (breakout at H4/L4)
# - Volume filter reduces false breakouts, ATR stop manages risk in volatile markets
# - Using 1d HTF for pivots reduces noise and increases reliability

name = "4h_1d_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14) for volatility and stoploss
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume > 1.8x 20-period average (volume confirmation)
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.8 * avg_volume_20)
    
    # Align 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_spike_4h[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 1d Camarilla pivot levels using previous 1d bar
            # Need at least 1 previous 1d bar
            if i < 6:  # Need at least 6 bars to get previous 1d bar (6x4h = 1d)
                signals[i] = 0.0
                continue
                
            # Get previous 1d bar's high, low, close
            prev_idx_1d = (i // 6) - 1  # 6x 4h bars = 1x 1d bar, subtract 1 for previous
            if prev_idx_1d < 0:
                signals[i] = 0.0
                continue
                
            # Get the actual previous 1d bar values
            ph = high_1d[prev_idx_1d]
            pl = low_1d[prev_idx_1d]
            pc = close_1d[prev_idx_1d]
            
            # Calculate Camarilla levels
            range_ = ph - pl
            if range_ <= 0:
                signals[i] = 0.0
                continue
                
            # Camarilla H4 and L4 levels (breakout levels)
            h4 = pc + (range_ * 1.1 / 2)
            l4 = pc - (range_ * 1.1 / 2)
            
            # Align H4/L4 to 4h timeframe (they're constant within the 1d bar)
            # Create arrays of H4/L4 values for each 1d bar, then align
            h4_array = np.full_like(high_1d, np.nan)
            l4_array = np.full_like(low_1d, np.nan)
            
            for j in range(len(high_1d)):
                ph_j = high_1d[j]
                pl_j = low_1d[j]
                pc_j = close_1d[j]
                range_j = ph_j - pl_j
                if range_j > 0:
                    h4_array[j] = pc_j + (range_j * 1.1 / 2)
                    l4_array[j] = pc_j - (range_j * 1.1 / 2)
            
            # Align to 4h
            h4_aligned = align_htf_to_ltf(prices, df_1d, h4_array)
            l4_aligned = align_htf_to_ltf(prices, df_1d, l4_array)
            
            if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Camarilla breakout with volume confirmation
            if (high[i] >= h4_aligned[i] and    # Break above H4
                volume_spike_4h[i]):            # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= l4_aligned[i] and    # Break below L4
                  volume_spike_4h[i]):           # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals