#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when price closes above 20-bar high with 1d ATR(14) > 1d ATR(50) and volume > 1.8x 20-bar average.
# Short when price closes below 20-bar low with same conditions.
# Exit when price crosses the 10-bar moving average in the opposite direction.
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# ATR regime ensures we only trade in expanding volatility regimes, avoiding low-volatility chop.
# Volume confirmation adds conviction to breakouts. 6h timeframe balances trade frequency and responsiveness.

name = "6h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    high_rolling = pd.Series(high).rolling(window=lookback, min_periods=lookback)
    low_rolling = pd.Series(low).rolling(window=lookback, min_periods=lookback)
    donchian_high = high_rolling.max().shift(1).values  # Previous 20-bar high
    donchian_low = low_rolling.min().shift(1).values    # Previous 20-bar low
    
    # Calculate 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d data (14-period and 50-period)
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: expanding volatility (short-term ATR > long-term ATR)
    atr_regime = atr_14 > atr_50
    
    # Align 1d ATR regime to 6h timeframe (wait for 1d bar to close)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(atr_regime_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above 20-bar high with ATR regime expanding and volume spike
            if (close[i] > donchian_high[i] and 
                atr_regime_aligned[i] > 0.5 and  # True when aligned
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below 20-bar low with ATR regime expanding and volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_regime_aligned[i] > 0.5 and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 10-period MA
            if close[i] < ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 10-period MA
            if close[i] > ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals