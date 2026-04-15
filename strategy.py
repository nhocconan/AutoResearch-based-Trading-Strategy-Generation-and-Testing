#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold) + price > 1d EMA50 + volume > 1.3x 20-period avg
# Short when Williams %R(14) crosses below -20 (overbought) + price < 1d EMA50 + volume > 1.3x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
# Williams %R identifies exhaustion points in ranging/mean-reverting markets.
# EMA50 filter ensures we trade with the daily trend, avoiding counter-trend whipsaws.
# Volume confirmation reduces false signals. Works in both bull (buy dips) and bear (sell rallies).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # === 1d Indicator: EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Indicator: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * ((highest_high - close) / denominator), -50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 50) + 5  # Williams %R(14) + EMA50(1d)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        vol_confirm = not np.isnan(vol_sma_20) and volume[i] > (vol_sma_20 * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_sma_20)):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold reversal)
        # 2. Price > 1d EMA50 (uptrend filter)
        # 3. Volume confirmation
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        if (williams_r_prev <= -80 and williams_r[i] > -80) and \
           (close[i] > ema50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought reversal)
        # 2. Price < 1d EMA50 (downtrend filter)
        # 3. Volume confirmation
        elif (williams_r_prev >= -20 and williams_r[i] < -20) and \
             (close[i] < ema50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR14_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0