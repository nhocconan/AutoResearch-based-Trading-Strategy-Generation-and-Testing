#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period avg
# Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Elder Ray measures market power; EMA34 filter ensures we trade with higher timeframe trend.
# Works in bull markets (strong Bull Power) and bear markets (strong Bear Power) by requiring alignment with 1d EMA34.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA(34) for trend filter ===
    close_1d = df_1d['close'].values
    ema_span = 34
    ema_1d = pd.Series(close_1d).ewm(span=ema_span, adjust=False, min_periods=ema_span).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) ===
    # EMA(13) for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 20) + 5  # EMA13(13) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (buying pressure)
        # 2. Bear Power < 0 (selling pressure weak)
        # 3. Price > 1d EMA34 (uptrend on higher timeframe)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and \
           (bear_power[i] < 0) and \
           (close[i] > ema_1d_aligned[i]) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bull Power < 0 (buying pressure weak)
        # 2. Bear Power > 0 (selling pressure)
        # 3. Price < 1d EMA34 (downtrend on higher timeframe)
        # 4. Volume confirmation
        elif (bull_power[i] < 0) and \
             (bear_power[i] > 0) and \
             (close[i] < ema_1d_aligned[i]) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA13_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0