#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(50) pullback strategy with 4h EMA(200) trend filter and volume confirmation
# Long when: price pulls back to EMA(50) in uptrend (close > 4h EMA200) with volume > 1.5x avg
# Short when: price pulls back to EMA(50) in downtrend (close < 4h EMA200) with volume > 1.5x avg
# Uses 4h EMA200 for robust trend filter reducing whipsaws in both bull and bear markets
# Volume filter (1.5x) targets ~20-40 trades/year to minimize fee drag on 1h timeframe
# EMA(50) pullback provides good risk-reward entries during trending markets

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # === 4h Indicator: EMA200 ===
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # === 1h EMA(50) for pullback entries ===
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 50, 20) + 5  # EMA200 + EMA50 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price pulls back to EMA(50) (close <= EMA50)
        # 2. 4h EMA200 uptrend (close > 4h EMA200)
        # 3. Volume confirmation
        if (close[i] <= ema_50[i]) and \
           (close[i] > ema_200_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price pulls back to EMA(50) (close >= EMA50)
        # 2. 4h EMA200 downtrend (close < 4h EMA200)
        # 3. Volume confirmation
        elif (close[i] >= ema_50[i]) and \
             (close[i] < ema_200_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA50_Pullback_4hEMA200_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0