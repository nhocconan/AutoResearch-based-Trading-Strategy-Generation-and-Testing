#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR trailing stop
# - Uses 12h Camarilla pivot levels (H4/L4) from prior 12h bar for breakout signals
# - Confirms with 1d volume > 2.0x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop from 12h: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work in ranging markets (mean reversion at S3/R3) and trending markets (breakout at H4/L4)
# - Volume filter reduces false breakouts, ATR stop manages risk in volatile markets

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
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
    volume_1d = df_1d['volume'].values
    
    # 1d Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20)
    
    # Align 1d volume spike to 12h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 12h price data
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
        if np.isnan(volume_spike_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high (using 12h ATR approximated from price range)
            # Calculate ATR from 12h data using True Range
            # We'll compute a simplified ATR using recent 12h price ranges
            if i >= 2:  # Need at least 2 bars for ATR calculation
                # Calculate True Range for recent bars
                tr1 = high[max(0, i-13):i+1] - low[max(0, i-13):i+1]
                tr2 = np.abs(high[max(0, i-13):i+1] - np.roll(close[max(0, i-13):i+1], 1))
                tr3 = np.abs(low[max(0, i-13):i+1] - np.roll(close[max(0, i-13):i+1], 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr[0] = tr[0]  # First bar TR is just high-low
                atr_approx = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
                
                if low[i] <= highest_since_entry - (2.5 * atr_approx):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if i >= 2:
                # Calculate True Range for recent bars
                tr1 = high[max(0, i-13):i+1] - low[max(0, i-13):i+1]
                tr2 = np.abs(high[max(0, i-13):i+1] - np.roll(close[max(0, i-13):i+1], 1))
                tr3 = np.abs(low[max(0, i-13):i+1] - np.roll(close[max(0, i-13):i+1], 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr[0] = tr[0]
                atr_approx = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
                
                if high[i] >= lowest_since_entry + (2.5 * atr_approx):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 12h Camarilla pivot levels using previous 12h bar
            # Need at least 1 previous 12h bar
            if i < 2:  # Need at least 2 bars to get previous 12h bar
                signals[i] = 0.0
                continue
                
            # Get previous 12h bar's high, low, close
            # Each 12h bar = 48 of 15m bars, but we're on 12h timeframe so each bar is 12h
            prev_idx = i - 1  # Previous 12h bar
            if prev_idx < 0:
                signals[i] = 0.0
                continue
                
            # Get the actual previous 12h bar values
            ph = high[prev_idx]
            pl = low[prev_idx]
            pc = close[prev_idx]
            
            # Calculate Camarilla levels
            range_ = ph - pl
            if range_ <= 0:
                signals[i] = 0.0
                continue
                
            # Camarilla H4 and L4 levels (breakout levels)
            h4 = pc + (range_ * 1.1 / 2)
            l4 = pc - (range_ * 1.1 / 2)
            
            # Look for Camarilla breakout with volume confirmation
            if (high[i] >= h4 and    # Break above H4
                volume_spike_1d_aligned[i]):   # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= l4 and    # Break below L4
                  volume_spike_1d_aligned[i]):  # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals