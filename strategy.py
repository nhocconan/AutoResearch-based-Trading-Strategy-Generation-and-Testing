#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when 6h Bull Power > 0 + 1d EMA34 rising + volume > 1.5x 20-period avg
# Short when 6h Bear Power < 0 + 1d EMA34 falling + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Elder Ray measures bull/bear power relative to EMA13. Combined with 1d EMA34 trend filter, it avoids counter-trend trades.
# Works in bull markets (buy strength on dips) and bear markets (sell weakness on rallies) by requiring alignment with higher timeframe trend.

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
    
    # === 1d Indicator: EMA34 (trend direction) ===
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA34 slope (rising/falling)
    ema34_slope = np.zeros_like(ema34_1d_aligned)
    ema34_slope[1:] = ema34_1d_aligned[1:] - ema34_1d_aligned[:-1]
    ema34_rising = ema34_slope > 0
    ema34_falling = ema34_slope < 0
    
    # === 6h Indicator: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13, 20) + 2  # EMA34(34) + EMA13(13) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 6h Bull Power > 0 (buying strength)
        # 2. 1d EMA34 rising (uptrend)
        # 3. Volume confirmation
        if (bull_power[i] > 0) and \
           ema34_rising[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. 6h Bear Power < 0 (selling pressure)
        # 2. 1d EMA34 falling (downtrend)
        # 3. Volume confirmation
        elif (bear_power[i] < 0) and \
             ema34_falling[i] and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA13_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0