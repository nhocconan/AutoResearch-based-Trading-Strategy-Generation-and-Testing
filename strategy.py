#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H3/L3 mean reversion with 1d volume confirmation and ATR trailing stop
# - Uses 1d Camarilla pivot levels (H3/L3) for mean reversion entries
# - Confirms with 4h volume > 1.5x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla H3/L3 work as mean reversion levels in ranging markets
# - Volume filter reduces false signals, ATR stop manages risk in volatile markets

name = "4h_1d_camarilla_h3l3_mr_v1"
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
    volume_4h = prices['volume'].values  # Use 4h volume for confirmation
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14) for volatility and stoploss
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 4h Volume > 1.5x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * avg_volume_20)
    
    # Align 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_4h.astype(float))
    
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
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike_4h_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 1d Camarilla pivot levels using previous 1d bar
            # Need at least 1 previous 1d bar
            if i < 6:  # Need at least 6 bars to get previous 1d bar (4h * 6 = 1d)
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
                
            # Camarilla H3 and L3 levels (mean reversion levels)
            h3 = pc + (range_ * 1.1 / 4)
            l3 = pc - (range_ * 1.1 / 4)
            
            # Align H3/L3 to 4h timeframe (they're constant within the 1d bar)
            # Create arrays of H3/L3 values for each 1d bar, then align
            h3_array = np.full_like(high_1d, np.nan)
            l3_array = np.full_like(low_1d, np.nan)
            
            for j in range(len(high_1d)):
                ph_j = high_1d[j]
                pl_j = low_1d[j]
                pc_j = close_1d[j]
                range_j = ph_j - pl_j
                if range_j > 0:
                    h3_array[j] = pc_j + (range_j * 1.1 / 4)
                    l3_array[j] = pc_j - (range_j * 1.1 / 4)
            
            # Align to 4h
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3_array)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3_array)
            
            if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
                np.isnan(volume_spike_4h_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Camarilla mean reversion with volume confirmation
            if (low[i] <= l3_aligned[i] and    # Price touches or breaks L3 (long signal)
                volume_spike_4h_aligned[i]):   # Volume confirmation
                position = 1
                entry_price = low[i]
                highest_since_entry = low[i]
                lowest_since_entry = low[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (high[i] >= h3_aligned[i] and   # Price touches or breaks H3 (short signal)
                  volume_spike_4h_aligned[i]):   # Volume confirmation
                position = -1
                entry_price = high[i]
                highest_since_entry = high[i]  # Initialize for longs
                lowest_since_entry = high[i]
                signals[i] = -0.25
    
    return signals