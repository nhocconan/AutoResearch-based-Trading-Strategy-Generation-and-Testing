#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with daily trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with price above daily EMA50 and volume > 1.5x average.
# Goes short when price breaks below 20-period low with price below daily EMA50 and volume > 1.5x average.
# Uses ATR-based stoploss and time-based exit to limit drawdown.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled risk.

name = "4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            bars_in_trade += 1
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            # Time-based exit: max 10 bars (~40 hours)
            elif bars_in_trade >= 10:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            bars_in_trade += 1
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            # Time-based exit: max 10 bars (~40 hours)
            elif bars_in_trade >= 10:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with strong volume confirmation
            if vol_strong[i]:
                # Long breakout: price breaks above 20-period high with price above daily EMA50
                if close[i] > high_roll[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_in_trade = 0
                # Short breakdown: price breaks below 20-period low with price below daily EMA50
                elif close[i] < low_roll[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_in_trade = 0
    
    return signals