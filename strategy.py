#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > 1d EMA34 AND volume > 1.5x 20-period avg
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA34 AND volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Elder Ray measures bull/bear strength relative to EMA13. 1d EMA34 filter ensures we trade with higher timeframe trend.
# Volume confirmation avoids false breakouts. Works in bull markets (strong buying pressure) and bear markets (strong selling pressure).

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
    
    # === 1d Indicator: EMA34 (trend filter) ===
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema13  # Bull Power: close - EMA13
    bear_power = low - ema13    # Bear Power: low - EMA13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13, 20) + 5  # EMA34(34) + EMA13(13) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) - indicates buying pressure
        # 2. Price > 1d EMA34 (uptrend on higher timeframe)
        # 3. Volume confirmation
        if (bull_power[i] > 0) and (bear_power[i] < 0) and \
           (close[i] > ema34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bull Power < 0 AND Bear Power > 0 - indicates selling pressure
        # 2. Price < 1d EMA34 (downtrend on higher timeframe)
        # 3. Volume confirmation
        elif (bull_power[i] < 0) and (bear_power[i] > 0) and \
             (close[i] < ema34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA13_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0