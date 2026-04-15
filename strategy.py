#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from oversold + price > 1d EMA50 + volume > 1.5x 20-period avg
# Short when Williams %R crosses below -20 from overbought + price < 1d EMA50 + volume > 1.5x 20-period avg
# Williams %R identifies exhaustion points in both bull and bear markets.
# 1d EMA50 provides strong trend filter to trade with higher timeframe momentum.
# Volume confirmation ensures breakouts have participation.
# Discrete sizing (0.25) controls drawdown and minimizes fee churn.
# Target: ~50-100 trades/year on 6h timeframe.

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
    
    # === Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where Highest High = max(high over lookback), Lowest Low = min(low over lookback)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, ((highest_high - close) / hh_ll) * -100, -50)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 50, 20) + 5  # Williams %R + EMA50 + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Williams %R levels
        williams_r_prev = williams_r[i-1]
        williams_r_curr = williams_r[i]
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (from oversold)
        # 2. Price > 1d EMA50 (uptrend filter)
        # 3. Volume confirmation
        if (williams_r_prev <= -80 and williams_r_curr > -80) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (from overbought)
        # 2. Price < 1d EMA50 (downtrend filter)
        # 3. Volume confirmation
        elif (williams_r_prev >= -20 and williams_r_curr < -20) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0