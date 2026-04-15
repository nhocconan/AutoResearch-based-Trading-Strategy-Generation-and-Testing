#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg
# Uses ATR(14) for dynamic position sizing and trailing stop (signal=0 when price retraces 2*ATR from extreme)
# Discrete position sizing (0.25) to minimize fee drag. Target: 20-40 trades/year per symbol.
# Works in bull markets (breakouts continuation) and bear markets (breakdown continuation).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Indicators: Donchian(20), Volume SMA(20), ATR(14) ===
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility-based stops
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Donchian(20) and ATR(14) need 20 periods
    
    # Track extreme prices for trailing stop
    long_extreme = np.full(n, np.nan)
    short_extreme = np.full(n, np.nan)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Update extremes for trailing stop
        if i == warmup:
            long_extreme[i] = close[i]
            short_extreme[i] = close[i]
        else:
            long_extreme[i] = long_extreme[i-1]
            short_extreme[i] = short_extreme[i-1]
        
        # === LONG LOGIC ===
        # Enter long: break above Donchian high + volume confirmation
        if (close[i] > donchian_high[i-1]) and vol_confirm:
            signals[i] = 0.25
            long_extreme[i] = close[i]  # reset extreme on new entry
        # Exit long: price retraces 2*ATR from extreme
        elif signals[i-1] > 0:  # currently long
            if close[i] < long_extreme[i] - 2.0 * atr[i]:
                signals[i] = 0.0  # stop and reverse/flat
                long_extreme[i] = close[i]
            else:
                signals[i] = signals[i-1]  # hold position
                long_extreme[i] = max(long_extreme[i], close[i])  # update extreme
        else:
            signals[i] = 0.0
        
        # === SHORT LOGIC ===
        # Enter short: break below Donchian low + volume confirmation
        if (close[i] < donchian_low[i-1]) and vol_confirm:
            signals[i] = -0.25
            short_extreme[i] = close[i]  # reset extreme on new entry
        # Exit short: price rallies 2*ATR from extreme
        elif signals[i-1] < 0:  # currently short
            if close[i] > short_extreme[i] + 2.0 * atr[i]:
                signals[i] = 0.0  # stop and reverse/flat
                short_extreme[i] = close[i]
            else:
                signals[i] = signals[i-1]  # hold position
                short_extreme[i] = min(short_extreme[i], close[i])  # update extreme
        else:
            # Only set to 0 if not already set by long logic
            if signals[i] == 0.0 and signals[i-1] >= 0:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0