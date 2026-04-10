#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and session filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 4h
# - Volume filter: 1d volume > 1.5x 20-period average volume (ensures participation)
# - Session filter: Trade only during 08-20 UTC (avoid low liquidity hours)
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(20) on 4h
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines

name = "1h_4h_1d_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 4h ATR(20) for stoploss
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 4h Camarilla levels (based on previous day's range)
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4, etc.
    # We use previous day's high/low to avoid look-ahead
    prev_high_4h = np.roll(high_4h, 1)  # Previous 4h bar's high
    prev_low_4h = np.roll(low_4h, 1)    # Previous 4h bar's low
    prev_close_4h = np.roll(close_4h, 1) # Previous 4h bar's close
    
    # Calculate Camarilla levels for each 4h bar using previous bar's range
    camarilla_high = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 4  # H3 level
    camarilla_low = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 4   # L3 level
    
    # Align Camarilla levels to 1h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reversion to Camarilla L3 OR stoploss hit
            if prices['close'].iloc[i] < camarilla_low_aligned[i] or \
               prices['close'].iloc[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price reversion to Camarilla H3 OR stoploss hit
            if prices['close'].iloc[i] > camarilla_high_aligned[i] or \
               prices['close'].iloc[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with volume filter
            if vol_spike_aligned[i]:
                # Long: price breaks above Camarilla H3
                if prices['close'].iloc[i] > camarilla_high_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L3
                elif prices['close'].iloc[i] < camarilla_low_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.20
    
    return signals