#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and increasing, Bear Power < 0 and decreasing, with 1d EMA50 uptrend and volume confirmation.
# Short when Bear Power < 0 and decreasing, Bull Power > 0 and increasing, with 1d EMA50 downtrend and volume confirmation.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Works in bull/bear: 1d EMA50 filters counter-trend trades, Elder Ray captures momentum shifts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Elder Ray momentum (change from previous bar)
        bull_power_mom = bull_power[i] - bull_power[i-1] if i > 0 else 0
        bear_power_mom = bear_power[i] - bear_power[i-1] if i > 0 else 0
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (bulls in control)
        # 2. Bull Power increasing (momentum building)
        # 3. Bear Power < 0 (bears weak)
        # 4. 1d EMA50 uptrend (close > EMA50)
        # 5. Volume confirmation
        if (bull_power[i] > 0 and
            bull_power_mom > 0 and
            bear_power[i] < 0 and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (bears in control)
        # 2. Bear Power decreasing (momentum building downward)
        # 3. Bull Power > 0 (bulls weak)
        # 4. 1d EMA50 downtrend (close < EMA50)
        # 5. Volume confirmation
        elif (bear_power[i] < 0 and
              bear_power_mom < 0 and
              bull_power[i] > 0 and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0