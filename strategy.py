#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume spike
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) with confirmation:
#    - 1d EMA50 uptrend (close > EMA50) 
#    - Volume > 2.0x 20-period volume SMA
# Short when Bear Power < 0 (close < EMA13) AND Bull Power < 0 (close < EMA13) with confirmation:
#    - 1d EMA50 downtrend (close < EMA50)
#    - Volume > 2.0x 20-period volume SMA
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Elder Ray measures bull/bear strength relative to EMA13, providing institutional-grade trend/strength filter.
# Volume threshold (2.0x) targets ~15-35 trades/year on 6h timeframe to avoid overtrading.

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
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13, 20) + 5  # EMA50 + EMA13 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (high > EMA13) AND Bear Power < 0 (low < EMA13) - price straddles EMA13 showing balance
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (bull_power[i] > 0) and (bear_power[i] < 0) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (low < EMA13) AND Bull Power < 0 (high < EMA13) - price below EMA13 showing bear control
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (bear_power[i] < 0) and (bull_power[i] < 0) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0