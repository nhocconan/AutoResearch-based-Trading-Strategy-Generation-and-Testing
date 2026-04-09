#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR trailing stop
# - Uses 4h Donchian channel (20-period) for breakout entries
# - Requires volume > 2.0 * 20-period volume average for confirmation (strict filter)
# - Uses ATR(14) for dynamic trailing stoploss (2.5 * ATR) and position sizing (0.25)
# - Works in bull markets via breakouts above upper channel, in bear via breakdowns below lower channel
# - Target: 15-30 trades/year on 4h timeframe (60-120 total over 4 years) to avoid fee drag
# - Donchian channels provide robust volatility-adaptive support/resistance levels

name = "4h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper channel: highest high of last 20 periods
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss and sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop or mean reversion to midpoint
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < (high_roll[i] + low_roll[i]) / 2:  # Mean reversion to midpoint
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop or mean reversion to midpoint
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > (high_roll[i] + low_roll[i]) / 2:  # Mean reversion to midpoint
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > high_roll[i] and volume_confirm[i]:  # Break above upper channel
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < low_roll[i] and volume_confirm[i]:  # Break below lower channel
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals