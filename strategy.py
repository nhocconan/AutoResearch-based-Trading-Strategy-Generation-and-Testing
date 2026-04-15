#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) on 6d? Wait, correction:
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Actually: Bull Power > 0 AND Bear Power < 0 implies High > EMA13 AND Low < EMA13 (price straddles EMA13) - not useful.
# Correct Elder Ray interpretation: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bull Power increasing (momentum up)
# Short when Bear Power < 0 AND Bear Power decreasing (momentum down)
# But simpler: use 1d EMA34 as trend filter, 6h for entry:
# Long when 6h Bull Power > 0 AND 1d EMA34 up (trend up) AND volume > 1.5x avg
# Short when 6h Bear Power < 0 AND 1d EMA34 down (trend down) AND volume > 1.5x avg
# Uses discrete sizing (0.25) for low turnover. Target: 50-150 trades over 4 years.

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
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray components ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13, 20) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (strong buying pressure)
        # 2. 1d EMA34 trending up (current > previous)
        # 3. Volume confirmation
        if (bull_power[i] > 0) and \
           (ema_1d_aligned[i] > ema_1d_aligned[i-1]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (strong selling pressure)
        # 2. 1d EMA34 trending down (current < previous)
        # 3. Volume confirmation
        elif (bear_power[i] < 0) and \
             (ema_1d_aligned[i] < ema_1d_aligned[i-1]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA13_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0