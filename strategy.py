#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Works in bull markets by capturing breakouts with the trend.
# Works in bear markets by requiring volume confirmation to avoid false breakdowns,
# and using 1w EMA50 to only take shorts when price is below weekly trend (bearish bias).
# Discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above weekly EMA50 (bullish bias from higher timeframe)
        # 2. Price breaks above daily Donchian(20) high with volume
        # 3. Volume confirmation: volume > 2.0x average (strict to reduce trades)
        if (close[i] > ema_50_1w_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 2.0):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA50 (bearish bias from higher timeframe)
        # 2. Price breaks below daily Donchian(20) low with volume
        # 3. Volume confirmation: volume > 2.0x average (strict to reduce trades)
        elif (close[i] < ema_50_1w_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 2.0):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0