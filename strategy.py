#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and ATR trailing stop
# - Uses weekly Donchian channel (20-period) for breakout signals
# - Confirms with 1d volume > 1.8x its 20-period average (strong participation) 
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)
# - Donchian breakouts capture strong trends, volume filter reduces false signals,
#   ATR stop manages risk in volatile markets. Works in both bull and bear markets
#   by capturing directional moves regardless of overall market direction.

name = "1d_1w_donchian_volume_atr_v1"
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
    
    # 1w Volume > 1.8x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (1.8 * avg_volume_20)
    
    # Align 1w indicators to 1d
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
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
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i]) or
            atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 1w Donchian channel levels using previous 1w bar
            # Need at least 1 previous 1w bar
            if i < 7:  # Need at least 7 days to get previous 1w bar
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
            
            # Calculate Donchian channel (20-period) for the previous 1w bar
            # We need 20 previous 1w bars to calculate the channel
            lookback_start = max(0, prev_idx_1w - 19)
            lookback_end = prev_idx_1w + 1
            
            if lookback_end - lookback_start < 20:
                signals[i] = 0.0
                continue
                
            # Calculate highest high and lowest low over the lookback period
            highest_high = np.max(high_1w[lookback_start:lookback_end])
            lowest_low = np.min(low_1w[lookback_start:lookback_end])
            
            # Align Donchian levels to 1d timeframe (they're constant within the 1w bar)
            # Create arrays of Donchian values for each 1w bar, then align
            upper_array = np.full_like(high_1w, np.nan)
            lower_array = np.full_like(low_1w, np.nan)
            
            for j in range(len(high_1w)):
                lookback_start_j = max(0, j - 19)
                lookback_end_j = j + 1
                if lookback_end_j - lookback_start_j >= 20:
                    upper_array[j] = np.max(high_1w[lookback_start_j:lookback_end_j])
                    lower_array[j] = np.min(low_1w[lookback_start_j:lookback_end_j])
            
            # Align to 1d
            upper_aligned = align_htf_to_ltf(prices, df_1w, upper_array)
            lower_aligned = align_htf_to_ltf(prices, df_1w, lower_array)
            
            if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
                np.isnan(volume_spike_1w_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Donchian breakout with volume confirmation
            if (high[i] >= upper_aligned[i] and    # Break above upper channel
                volume_spike_1w_aligned[i]):       # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= lower_aligned[i] and    # Break below lower channel
                  volume_spike_1w_aligned[i]):      # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals