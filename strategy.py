#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w EMA34 trend filter + volume confirmation
# Long when Bull Power > 0, Bear Power < 0, price > 1w EMA34, volume > 1.5x 20-period avg
# Short when Bull Power < 0, Bear Power > 0, price < 1w EMA34, volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1w EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~50-150 trades over 4 years (~12-37/year) to minimize fee drag on 6h timeframe.
# Elder Ray measures bull/bear power relative to EMA13, providing institutional-grade trend strength.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w Indicator: EMA34 ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Elder Ray Indicators (Bull Power, Bear Power) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    # Using 13-period EMA as standard for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13, 20) + 5  # EMA34(1w) + EMA13 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (bulls in control)
        # 2. Bear Power < 0 (bears weak)
        # 3. Price > 1w EMA34 (primary trend up)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and (bear_power[i] < 0) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bull Power < 0 (bulls weak)
        # 2. Bear Power > 0 (bears in control)
        # 3. Price < 1w EMA34 (primary trend down)
        # 4. Volume confirmation
        elif (bull_power[i] < 0) and (bear_power[i] > 0) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0