#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h Camarilla pivots
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and levels
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels (standard multipliers)
    r4_12h = close_12h + range_12h * 1.1 / 2
    s4_12h = close_12h - range_12h * 1.1 / 2
    
    # Align 12h pivots to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # ATR for dynamic exit
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ma_30[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        r4 = r4_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_30[i]
        
        # Breakout signals
        long_signal = price_close > r4 and volume_confirmed
        short_signal = price_close < s4 and volume_confirmed
        
        # Exit conditions: trailing stop at 2x ATR from extreme
        if position == 1:
            # Track highest high since entry
            if i == 100:
                highest_high = price_close
            else:
                highest_high = max(highest_high, price_close)
            exit_long = price_close < (highest_high - 2 * atr[i])
        elif position == -1:
            # Track lowest low since entry
            if i == 100:
                lowest_low = price_close
            else:
                lowest_low = min(lowest_low, price_close)
            exit_short = price_close > (lowest_low + 2 * atr[i])
        else:
            exit_long = exit_short = False
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Camarilla breakout strategy with volume confirmation and ATR trailing stop.
# Enters long when price breaks above 12h R4 with volume confirmation (>1.5x 30-period average volume).
# Enters short when price breaks below 12h S4 with volume confirmation.
# Exits when price retraces 2x ATR from the session extreme (highest high for longs, lowest low for shorts).
# Uses volume confirmation to ensure institutional participation and reduce false breakouts.
# ATR trailing stop adapts to volatility, capturing trends while limiting losses in choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee efficiency.
# Works in both bull and bear markets by trading breakouts in either direction with volatility-adjusted exits.