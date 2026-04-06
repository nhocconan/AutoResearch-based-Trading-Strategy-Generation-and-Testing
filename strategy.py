#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian upper band (20) with 12h uptrend and volume spike
# Short when price breaks below Donchian lower band (20) with 12h downtrend and volume spike
# Uses volume spike (2x 20-period average) to confirm breakouts and avoid false signals
# Target: 75-150 total trades over 4 years with controlled risk (max 0.30 position)

name = "6h_donchian20_12h_ema50_vol_v1"
timeframe = "6h"
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
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # EMA50 calculation on 12h
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower band or trend changes
            elif close[i] < lowest_low[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper band or trend changes
            elif close[i] > highest_high[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for breakout entries with volume confirmation
            # Long: price breaks above Donchian upper band, uptrend, volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower band, downtrend, volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals