#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and ATR trailing stop
# - Uses 1h Camarilla pivot levels (H4/L4) for breakout signals
# - Confirms with 4h volume > 1.8x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.20 (20% of capital) to manage drawdown in bear markets
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - 1h timeframe balances responsiveness with reasonable trade frequency
# - Volume filter reduces false breakouts, ATR stop manages risk in volatile markets
# - Works in both bull/bear: breakouts capture trends, tight stops limit losses

name = "1h_4h_camarilla_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h True Range for ATR
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr_4h[0]
    
    # 4h ATR(14) for volatility and stoploss
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume > 1.8x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.8 * avg_volume_20)
    
    # Align 4h indicators to 1h
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(atr_4h_aligned[i]) or 
            np.isnan(volume_spike_4h_aligned[i]) or
            atr_4h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Calculate 1h Camarilla pivot levels using previous 1h bar
            # Need at least 1 previous 1h bar
            if i < 4:  # Need at least 4 bars to get previous 1h bar (for safety)
                signals[i] = 0.0
                continue
                
            # Get previous 1h bar's high, low, close
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            
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
                volume_spike_4h_aligned[i]):  # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.20
            elif (low[i] <= l4 and    # Break below L4
                  volume_spike_4h_aligned[i]):  # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.20
    
    return signals