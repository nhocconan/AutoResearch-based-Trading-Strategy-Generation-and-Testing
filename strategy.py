#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter
# Long when: Alligator bullish (jaw < teeth < lips), Bull Power > 0, price > 12h EMA50
# Short when: Alligator bearish (jaw > teeth > lips), Bear Power < 0, price < 12h EMA50
# Uses Williams Alligator (SMAs: jaw=13, teeth=8, lips=5) to identify trend direction and avoid whipsaws.
# Elder Ray measures bull/bear power relative to EMA13 to confirm trend strength.
# 12h EMA50 acts as higher-timeframe trend filter to align with dominant trend.
# Designed for low-frequency entries (~20-40 trades/year) to minimize fee drag on 6h timeframe.
# Works in bull markets via trend continuation and bear markets via counter-trend retracements to EMA13.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: EMA50 ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Williams Alligator (SMAs) ===
    # Jaw: 13-period SMMA (smoothed with 8-period offset)
    # Teeth: 8-period SMMA (smoothed with 5-period offset)
    # Lips: 5-period SMMA (smoothed with 3-period offset)
    # Using EMA as proxy for SMMA with same period for simplicity and responsiveness
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # === Elder Ray: Bull Power and Bear Power ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 50) + 5  # Alligator + EMA50 + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: jaw < teeth < lips
        # 2. Bull Power > 0 (bulls in control)
        # 3. Price > 12h EMA50 (higher-timeframe uptrend)
        if (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and \
           (bull_power[i] > 0) and (close[i] > ema_50_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: jaw > teeth > lips
        # 2. Bear Power < 0 (bears in control)
        # 3. Price < 12h EMA50 (higher-timeframe downtrend)
        elif (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and \
             (bear_power[i] < 0) and (close[i] < ema_50_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_ElderRay_12hEMA50_v1"
timeframe = "6h"
leverage = 1.0