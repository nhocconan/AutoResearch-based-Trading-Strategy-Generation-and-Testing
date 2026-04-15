#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume spike confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > 1d EMA34 AND volume > 2.0x 20-period volume SMA
# Short when Bear Power < 0 (low < EMA13) AND Bull Power < 0 (close < EMA13) AND price < 1d EMA34 AND volume > 2.0x 20-period volume SMA
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Elder Ray measures bull/bear power relative to EMA13, capturing momentum exhaustion. 1d EMA34 ensures we trade with the higher timeframe trend.
# Volume spike confirms institutional participation. Works in bull markets (strong bull power) and bear markets (strong bear power).

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
    
    # === 6h Indicator: EMA13 for Elder Ray calculation ===
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicator: Volume SMA for confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 (trend filter) ===
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = close[i] - ema13[i]  # Bull Power = Close - EMA13
        bear_power = low[i] - ema13[i]    # Bear Power = Low - EMA13
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (close above EMA13)
        # 2. Bear Power < 0 (low below EMA13) - ensures we're not in extreme overbought
        # 3. Price above 1d EMA34 (uptrend on higher timeframe)
        # 4. Volume confirmation
        if (bull_power > 0) and (bear_power < 0) and \
           (close[i] > ema34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (low below EMA13)
        # 2. Bull Power < 0 (close below EMA13) - ensures we're not in extreme oversold
        # 3. Price below 1d EMA34 (downtrend on higher timeframe)
        # 4. Volume confirmation
        elif (bear_power < 0) and (bull_power < 0) and \
             (close[i] < ema34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0