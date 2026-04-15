#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# Long when Williams %R(14) < -80 (oversold) + 1d EMA50 > EMA200 (bullish trend) + volume > 1.5x 20-period avg
# Short when Williams %R(14) > -20 (overbought) + 1d EMA50 < EMA200 (bearish trend) + volume > 1.5x 20-period avg
# Williams %R captures short-term extremes, EMA crossover filters for trend direction to avoid counter-trend whipsaws.
# Volume confirmation ensures breakout validity. Designed for low trade frequency (15-30/year) to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) by requiring trend alignment.

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
    
    # === 1d Indicator: EMA50 and EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Trend: bullish when EMA50 > EMA200, bearish when EMA50 < EMA200
    ema50_gt_ema200 = ema50_1d > ema200_1d
    ema50_lt_ema200 = ema50_1d < ema200_1d
    ema50_gt_ema200_aligned = align_htf_to_ltf(prices, df_1d, ema50_gt_ema200)
    ema50_lt_ema200_aligned = align_htf_to_ltf(prices, df_1d, ema50_lt_ema200)
    
    # === 4h Indicator: Williams %R (14-period) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 200) + 20  # Williams %R(14) + EMA200(1d) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_gt_ema200_aligned[i]) or
            np.isnan(ema50_lt_ema200_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Bullish trend (1d EMA50 > EMA200)
        # 3. Volume confirmation
        if (williams_r[i] < -80) and \
           (ema50_gt_ema200_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Bearish trend (1d EMA50 < EMA200)
        # 3. Volume confirmation
        elif (williams_r[i] > -20) and \
             (ema50_lt_ema200_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR14_1dEMA50_200_Trend_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0