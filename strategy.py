#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Goes long when price breaks above 20-day high with volume > 1.3x average and price above 1w EMA.
# Goes short when price breaks below 20-day low with volume > 1.3x average and price below 1w EMA.
# Exits when price reverses to touch 10-day EMA or stoploss hit (2*ATR).
# Trend filter ensures alignment with higher timeframe direction to avoid counter-trend trades.
# Target: 30-100 total trades over 4 years (7-25/year) with controlled risk.

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d Donchian channels (20-period)
    # Using rolling window on daily data
    high_1d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_1d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit signal
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)  # Volume above 1.3x average
    
    # ATR for stoploss (14-period)
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
    
    for i in range(50, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(ema_10[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-day EMA or breaks below 20-day low
            elif close[i] < ema_10[i] or close[i] < low_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-day EMA or breaks above 20-day high
            elif close[i] > ema_10[i] or close[i] > high_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if vol_filter[i]:
                # Long breakout: price breaks above 20-day high with volume and above 1w EMA
                if close[i] > high_1d[i] and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below 20-day low with volume and below 1w EMA
                elif close[i] < low_1d[i] and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals