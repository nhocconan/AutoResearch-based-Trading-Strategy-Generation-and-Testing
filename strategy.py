#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ATR trailing stop
# - Uses 12h Camarilla pivot levels (H4/L4) for breakout signals
# - Confirms with 4h volume > 2.0x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivots work in ranging markets (mean reversion at S3/R3) and trending markets (breakout at H4/L4)
# - Volume filter reduces false breakouts, ATR stop manages risk in volatile markets

name = "4h_12h_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    # 12h ATR(14) for volatility and stoploss
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * avg_volume_20)
    
    # Align 12h indicators to 4h
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
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
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(volume_spike_12h_aligned[i]) or
            atr_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 12h Camarilla pivot levels using previous 12h bar
            # Need at least 1 previous 12h bar
            if i < 2:  # Need at least 2 bars to get previous 12h bar
                signals[i] = 0.0
                continue
                
            # Get previous 12h bar's high, low, close
            prev_idx_12h = (i // 3) - 1  # 3x 4h bars = 1x 12h bar, subtract 1 for previous
            if prev_idx_12h < 0:
                signals[i] = 0.0
                continue
                
            # Get the actual previous 12h bar values
            ph = high_12h[prev_idx_12h]
            pl = low_12h[prev_idx_12h]
            pc = close_12h[prev_idx_12h]
            
            # Calculate Camarilla levels
            range_ = ph - pl
            if range_ <= 0:
                signals[i] = 0.0
                continue
                
            # Camarilla H4 and L4 levels (breakout levels)
            h4 = pc + (range_ * 1.1 / 2)
            l4 = pc - (range_ * 1.1 / 2)
            
            # Align H4/L4 to 4h timeframe (they're constant within the 12h bar)
            # Create arrays of H4/L4 values for each 12h bar, then align
            h4_array = np.full_like(high_12h, np.nan)
            l4_array = np.full_like(low_12h, np.nan)
            
            for j in range(len(high_12h)):
                ph_j = high_12h[j]
                pl_j = low_12h[j]
                pc_j = close_12h[j]
                range_j = ph_j - pl_j
                if range_j > 0:
                    h4_array[j] = pc_j + (range_j * 1.1 / 2)
                    l4_array[j] = pc_j - (range_j * 1.1 / 2)
            
            # Align to 4h
            h4_aligned = align_htf_to_ltf(prices, df_12h, h4_array)
            l4_aligned = align_htf_to_ltf(prices, df_12h, l4_array)
            
            if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
                np.isnan(volume_spike_12h_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Camarilla breakout with volume confirmation
            if (high[i] >= h4_aligned[i] and    # Break above H4
                volume_spike_12h_aligned[i]):   # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= l4_aligned[i] and    # Break below L4
                  volume_spike_12h_aligned[i]):  # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals