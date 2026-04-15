#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA34) AND 1d EMA50 > EMA200 (uptrend) AND volume > 1.5x 20-period avg
# Short when Bear Power < 0 (close < EMA13) AND Bull Power > 0 (close > EMA34) AND 1d EMA50 < EMA200 (downtrend) AND volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Elder Ray measures bull/bear power relative to EMAs, providing clear trend strength signals.
# Works in bull markets (strong bull power) and bear markets (strong bear power) by requiring 1d trend alignment.

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
    
    # === 1d Indicators: EMA50 and EMA200 (trend filter) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Indicators: EMA13 and EMA34 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Bull Power = close - EMA13
    bull_power = close - ema13
    # Bear Power = close - EMA34
    bear_power = close - ema34
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 200) + 20  # EMA34 + EMA200(1d) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema34[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA34) = strong bullish momentum
        # 2. 1d Uptrend (EMA50 > EMA200)
        # 3. Volume confirmation
        if (bull_power[i] > 0) and (bear_power[i] < 0) and \
           (ema50_aligned[i] > ema200_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (close < EMA13) AND Bull Power > 0 (close > EMA34) = strong bearish momentum
        # 2. 1d Downtrend (EMA50 < EMA200)
        # 3. Volume confirmation
        elif (bear_power[i] < 0) and (bull_power[i] > 0) and \
             (ema50_aligned[i] < ema200_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA13_EMA34_1dEMA50_200_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0