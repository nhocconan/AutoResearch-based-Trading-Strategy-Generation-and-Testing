#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Long when: price breaks above 20-period Donchian high, 1d ATR(14) > 1.2 * 20-period ATR(14) MA, and volume > 1.5x 20-period volume MA
# Short when: price breaks below 20-period Donchian low, 1d ATR(14) > 1.2 * 20-period ATR(14) MA, and volume > 1.5x 20-period volume MA
# Exit when price returns to the opposite Donchian level (mean reversion) or opposite breakout
# Uses Donchian channels for structure, effective in trending markets (bull/bear) with volatility filter to avoid chop
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate ATR on 4h for stoploss reference (not used in entry, but for regime)
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = high[0] - low[0]  # First bar TR
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.zeros(n)
    
    # Get 1d data ONCE before loop for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        tr1_1d = high_1d - low_1d
        tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
        tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
        tr1_1d[0] = high_1d[0] - low_1d[0]
        tr2_1d[0] = np.abs(high_1d[0] - close_1d[0])
        tr3_1d[0] = np.abs(low_1d[0] - close_1d[0])
        tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
        atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    else:
        atr_14_1d = np.full(len(close_1d), np.nan)
    
    # Calculate 20-period MA of 1d ATR(14) for volatility regime filter
    if len(atr_14_1d) >= 20:
        atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
        # Volatility filter: current 1d ATR > 1.2 * 20-period MA (avoid low volatility/chop)
        vol_regime_filter = atr_14_1d > (1.2 * atr_ma_20_1d)
    else:
        vol_regime_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d volatility filter to 4h timeframe
    vol_regime_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_filter)
    
    # Calculate Donchian channels on 4h (20-period)
    if len(high) >= 20 and len(low) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_regime_filter_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volatility filter, and volume confirmation
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # Ensure breakout happens on this bar
                vol_regime_filter_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volatility filter, and volume confirmation
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # Ensure breakdown happens on this bar
                  vol_regime_filter_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low (mean reversion) or breaks below Donchian low (reversal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high (mean reversion) or breaks above Donchian high (reversal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals