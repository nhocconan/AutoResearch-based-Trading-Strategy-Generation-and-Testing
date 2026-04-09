#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ATR trailing stop
# - Uses 1w Camarilla pivot levels (H4/L4) for breakout signals
# - Confirms with 1d volume > 2.0x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 3.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)
# - Camarilla pivots work in ranging markets (mean reversion at S3/R3) and trending markets (breakout at H4/L4)
# - Volume filter reduces false breakouts, ATR stop manages risk in volatile markets
# - Using 1w HTF for pivots reduces noise and increases reliability for daily timeframe

name = "1d_1w_camarilla_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w True Range for ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr_1w[0]
    
    # 1w ATR(14) for volatility and stoploss
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume > 2.0x 20-period average (volume confirmation)
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20)
    
    # Align 1w indicators to 1d
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 1d price data
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
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(volume_spike_1d[i]) or
            atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 3.0x ATR from high
            if low[i] <= highest_since_entry - (3.0 * atr_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 3.0x ATR from low
            if high[i] >= lowest_since_entry + (3.0 * atr_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 1w Camarilla pivot levels using previous 1w bar
            # Need at least 1 previous 1w bar
            if i < 7:  # Need at least 7 bars to get previous 1w bar (7x1d = 1w)
                signals[i] = 0.0
                continue
                
            # Get previous 1w bar's high, low, close
            prev_idx_1w = (i // 7) - 1  # 7x 1d bars = 1x 1w bar, subtract 1 for previous
            if prev_idx_1w < 0:
                signals[i] = 0.0
                continue
                
            # Get the actual previous 1w bar values
            ph = high_1w[prev_idx_1w]
            pl = low_1w[prev_idx_1w]
            pc = close_1w[prev_idx_1w]
            
            # Calculate Camarilla levels
            range_ = ph - pl
            if range_ <= 0:
                signals[i] = 0.0
                continue
                
            # Camarilla H4 and L4 levels (breakout levels)
            h4 = pc + (range_ * 1.1 / 2)
            l4 = pc - (range_ * 1.1 / 2)
            
            # Align H4/L4 to 1d timeframe (they're constant within the 1w bar)
            # Create arrays of H4/L4 values for each 1w bar, then align
            h4_array = np.full_like(high_1w, np.nan)
            l4_array = np.full_like(low_1w, np.nan)
            
            for j in range(len(high_1w)):
                ph_j = high_1w[j]
                pl_j = low_1w[j]
                pc_j = close_1w[j]
                range_j = ph_j - pl_j
                if range_j > 0:
                    h4_array[j] = pc_j + (range_j * 1.1 / 2)
                    l4_array[j] = pc_j - (range_j * 1.1 / 2)
            
            # Align to 1d
            h4_aligned = align_htf_to_ltf(prices, df_1w, h4_array)
            l4_aligned = align_htf_to_ltf(prices, df_1w, l4_array)
            
            if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Camarilla breakout with volume confirmation
            if (high[i] >= h4_aligned[i] and    # Break above H4
                volume_spike_1d[i]):            # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= l4_aligned[i] and    # Break below L4
                  volume_spike_1d[i]):           # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals