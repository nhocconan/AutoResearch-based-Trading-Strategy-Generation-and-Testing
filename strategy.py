#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Bull Power = High - EMA34(close), Bear Power = Low - EMA34(close)
# Long when Bull Power > 0 AND Bear Power rising (less negative) + 1d EMA34 up + volume > 1.5x 20-period avg
# Short when Bear Power < 0 AND Bull Power falling (less positive) + 1d EMA34 down + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Elder Ray measures bull/bear strength relative to trend.
# Works in bull markets (strong bull power) and bear markets (strong bear power) by requiring EMA34 alignment.

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
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicator: Elder Ray (Bull/Bear Power) ===
    # Calculate EMA13 for power calculation (standard for Elder Ray)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Calculate 2-period change for power momentum
    bull_power_change = bull_power - np.roll(bull_power, 2)
    bear_power_change = bear_power - np.roll(bear_power, 2)
    bull_power_change[:2] = 0
    bear_power_change[:2] = 0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13, 20) + 2  # EMA34(1d) + EMA13(6h) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (strong bullish momentum)
        # 2. Bear Power rising (less negative = weakening bearish pressure)
        # 3. 1d EMA34 trending up (current > previous)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and \
           (bear_power_change[i] > 0) and \
           (i > 0 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (strong bearish momentum)
        # 2. Bull Power falling (less positive = weakening bullish pressure)
        # 3. 1d EMA34 trending down (current < previous)
        # 4. Volume confirmation
        elif (bear_power[i] < 0) and \
             (bull_power_change[i] < 0) and \
             (i > 0 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0