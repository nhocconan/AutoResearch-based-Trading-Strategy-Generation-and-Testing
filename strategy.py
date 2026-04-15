#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w EMA50 trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) + price > 1w EMA50 + volume > 1.5x 20-period avg
# Short when Bear Power < 0 AND Bull Power falling (less positive) + price < 1w EMA50 + volume > 1.5x 20-period avg
# Uses 1w HTF for major trend alignment, 6h for Elder Ray calculation and entry timing
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing trending moves
# Works in both bull and bear markets by requiring 1w EMA50 alignment and volume confirmation

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
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w Indicator: EMA50 ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    # We need 6h data for Elder Ray - use the prices dataframe directly since timeframe is 6h
    # Calculate EMA13 on 6h close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate rate of change for Elder Ray components (to detect rising/falling)
    bull_power_roc = pd.Series(bull_power).diff(periods=3).values  # 3-bar ROC
    bear_power_roc = pd.Series(bear_power).diff(periods=3).values  # 3-bar ROC
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_roc[i]) or np.isnan(bear_power_roc[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (buying pressure)
        # 2. Bear Power rising (less negative = weakening selling pressure)
        # 3. Price above 1w EMA50 (major uptrend)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and (bear_power_roc[i] > 0) and \
           (close[i] > ema_50_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (selling pressure)
        # 2. Bull Power falling (less positive = weakening buying pressure)
        # 3. Price below 1w EMA50 (major downtrend)
        # 4. Volume confirmation
        elif (bear_power[i] < 0) and (bull_power_roc[i] < 0) and \
             (close[i] < ema_50_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0