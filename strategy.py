#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume spike
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND 1d EMA34 up AND volume > 2x 20-period avg
# Short when Bear Power < 0 (low < EMA13) AND Bull Power < 0 (close < EMA13) AND 1d EMA34 down AND volume > 2x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
# Elder Ray measures bull/bear power relative to EMA13. 1d EMA34 filter ensures we trade with higher timeframe trend.
# Volume spike confirms institutional participation. Works in bull trends (buy strength) and bear trends (sell weakness).

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
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13  # > 0 indicates bull power
    bear_power = low - ema_13    # < 0 indicates bear power
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (close > EMA13) - bulls in control
        # 2. Bear Power < 0 (low < EMA13) - bears not strong enough to push low below EMA13
        # 3. 1d EMA34 trending up (current > previous)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and (bear_power[i] < 0) and \
           (i > 0 and ema_34_aligned[i] > ema_34_aligned[i-1]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (low < EMA13) - bears in control
        # 2. Bull Power < 0 (close < EMA13) - bulls not strong enough to push close above EMA13
        # 3. 1d EMA34 trending down (current < previous)
        # 4. Volume confirmation
        elif (bear_power[i] < 0) and (bull_power[i] < 0) and \
             (i > 0 and ema_34_aligned[i] < ema_34_aligned[i-1]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0