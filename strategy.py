#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + daily volume confirmation + daily ATR filter
# Go long when price breaks above 12h Donchian upper band (20) with volume > 1.5x daily average
# Go short when price breaks below 12h Donchian lower band (20) with volume > 1.5x daily average
# Use daily ATR(14) to filter volatility and set stoploss at 2x ATR
# Target: 50-150 total trades over 4 years with low frequency to avoid fee drag

name = "12h_donchian20_1d_vol_atr_v1"
timeframe = "12h"
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
    
    # 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 12h timeframe (no shift needed as we use same timeframe)
    donch_high_aligned = donch_high  # Already aligned
    donch_low_aligned = donch_low    # Already aligned
    
    # Daily data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily volume average (20-period)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily data to 12h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below lower Donchian band
            elif close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above upper Donchian band
            elif close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above upper Donchian band + volume spike
            if (close[i] > donch_high_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower Donchian band + volume spike
            elif (close[i] < donch_low_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals