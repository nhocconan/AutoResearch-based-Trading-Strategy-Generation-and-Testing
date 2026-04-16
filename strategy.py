#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and volume exhaustion
# Long when: price touches lower Bollinger Band (20,2) + 4h EMA50 uptrend + volume spike > 1.8x 20-period avg
# Short when: price touches upper Bollinger Band (20,2) + 4h EMA50 downtrend + volume spike > 1.8x 20-period avg
# Uses 1h only for entry timing, 4h for trend direction to avoid whipsaws.
# Volume spike identifies exhaustion moves likely to reverse.
# Bollinger Bands provide dynamic support/resistance that adapts to volatility.
# Session filter (08-20 UTC) reduces noise. Discrete size 0.20 controls drawdown and fee drag.

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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA50 ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h Bollinger Bands (20,2) ===
    # Middle = SMA20, Upper = Middle + 2*StdDev, Lower = Middle - 2*StdDev
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    bollinger_upper = sma_20 + (2 * std_20)
    bollinger_lower = sma_20 - (2 * std_20)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50 + Bollinger(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA (exhaustion move)
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # === LONG CONDITIONS ===
        # 1. Price touches or breaks below lower Bollinger Band (low <= lower)
        # 2. 4h EMA50 uptrend (close > EMA50)
        # 3. Volume exhaustion spike
        if (low[i] <= bollinger_lower[i]) and \
           (close[i] > ema_50_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price touches or breaks above upper Bollinger Band (high >= upper)
        # 2. 4h EMA50 downtrend (close < EMA50)
        # 3. Volume exhaustion spike
        elif (high[i] >= bollinger_upper[i]) and \
             (close[i] < ema_50_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Bollinger20_4hEMA50_VolumeExhaustion_v1"
timeframe = "1h"
leverage = 1.0