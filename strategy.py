#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and ATR trailing stop
# - Uses 6h Donchian channel breakout for trend following entries
# - Requires 12h volume > 1.5 * 24-period volume average for confirmation (balanced filter)
# - Uses ATR(14) for dynamic trailing stoploss (2.5 * ATR) and position sizing (0.25)
# - Works in bull markets via breakouts above upper channel, in bear via breakdowns below lower channel
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to avoid fee drag
# - Donchian channels provide adaptive volatility-based structure that works in all regimes

name = "6h_12h_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 24:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: volume > 1.5 * 24-period average
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_confirm_12h = volume_12h > (1.5 * vol_ma_12h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm_12h)
    
    # Pre-compute 6h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h ATR(14) for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop or Donchian mean reversion
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < donch_low[i]:  # Mean reversion exit (break below lower channel)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop or Donchian mean reversion
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > donch_high[i]:  # Mean reversion exit (break above upper channel)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > donch_high[i] and volume_confirm_aligned[i]:  # Break above upper channel
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < donch_low[i] and volume_confirm_aligned[i]:  # Break below lower channel
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals