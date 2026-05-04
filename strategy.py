#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses 1d Donchian channel breakouts for structure - captures strong momentum at key support/resistance
# 1w EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.5x average volume) - tighter to reduce trades to 7-25/year target
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Donchian levels (20-period) from prior completed 1d bar
    # Donchian Upper = highest high of last 20 periods
    # Donchian Lower = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND price > 1w EMA34 AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND price < 1w EMA34 AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian OR price crosses below 1w EMA34
            if close[i] < donchian_lower[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian OR price crosses above 1w EMA34
            if close[i] > donchian_upper[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals