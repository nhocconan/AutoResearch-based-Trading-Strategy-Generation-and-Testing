#!/usr/bin/env python3
"""
4h_VWAP_MeanReversion_With_Trend_Filter
Hypothesis: Mean reversion to VWAP on 4h timeframe, filtered by daily trend (EMA50) and volume spike.
Works in both bull and bear markets by only taking reversions in direction of higher timeframe trend.
VWAP acts as dynamic support/resistance, with volume confirming institutional interest.
Target: 20-30 trades/year, avoiding overtrading while capturing mean reversion moves.
"""

name = "4h_VWAP_MeanReversion_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, 0)
    
    # Volume filter: current volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    # Price deviation from VWAP as percentage
    price_dev = (close - vwap) / vwap * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA50 (50 bars) and VWAP calculation
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema_50_daily_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below VWAP (oversold) AND uptrend (above daily EMA50) AND volume spike
            if price_dev[i] < -1.5 and close[i] > ema_50_daily_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (overbought) AND downtrend (below daily EMA50) AND volume spike
            elif price_dev[i] > 1.5 and close[i] < ema_50_daily_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above VWAP OR trend turns bearish
            if close[i] > vwap[i] or close[i] < ema_50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below VWAP OR trend turns bullish
            if close[i] < vwap[i] or close[i] > ema_50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals