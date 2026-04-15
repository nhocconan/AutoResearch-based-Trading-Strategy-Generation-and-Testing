#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike
# Long when Alligator jaws < teeth < lips (bullish alignment) + price > lips + 1d EMA50 uptrend + volume > 2.0x 20-period avg
# Short when Alligator jaws > teeth > lips (bearish alignment) + price < lips + 1d EMA50 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams Alligator provides trend identification with built-in smoothing reducing whipsaws.
# Volume threshold (2.0x) targets ~20-40 trades/year on 4h timeframe to avoid overtrading.
# 1d EMA50 provides strong multi-timeframe trend filter aligning with higher timeframe momentum.

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
    
    # === Williams Alligator (13,8,5) ===
    # Jaw (13-period SMMA, 8 bars ahead)
    # Teeth (8-period SMMA, 5 bars ahead)
    # Lips (5-period SMMA, 3 bars ahead)
    # Using SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + values[i]) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13, 20) + 5  # EMA50 + Alligator + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: jaw < teeth < lips
        # 2. Price > lips (green line)
        # 3. 1d EMA50 uptrend (close > EMA50)
        # 4. Volume confirmation
        if (jaw[i] < teeth[i] < lips[i]) and \
           (close[i] > lips[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: jaw > teeth > lips
        # 2. Price < lips (green line)
        # 3. 1d EMA50 downtrend (close < EMA50)
        # 4. Volume confirmation
        elif (jaw[i] > teeth[i] > lips[i]) and \
             (close[i] < lips[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Williams_Alligator_1dEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0