#!/usr/bin/env python3
"""
4h_1d_4H_Trend_1D_Pullback_Scalper
Hypothesis: Trade pullbacks to the 4h EMA20 in the direction of the daily trend (EMA50). Enter on bullish/bearish engulfing candles with volume confirmation. Works in bull markets via trend continuation and in bear markets via shorting rallies against the daily downtrend. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Daily EMA50 for trend
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # Candlestick patterns: bullish/bearish engulfing
    bullish_engulf = (close > open_) & (open_ < close_prev) & (close > close_prev) & (open_ < close_prev)
    bearish_engulf = (close < open_) & (open_ > close_prev) & (close < close_prev) & (open_ > close_prev)
    # Need previous bar data
    open_ = prices['open'].values
    close_prev = np.roll(close, 1)
    open_prev = np.roll(open_, 1)
    close_prev[0] = np.nan
    open_prev[0] = np.nan
    bullish_engulf = (close > open_) & (open_ < close_prev) & (close > close_prev) & (open_ < close_prev)
    bearish_engulf = (close < open_) & (open_ > close_prev) & (close < close_prev) & (open_ > close_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(ema50_daily_aligned[i]) or np.isnan(ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_daily_aligned[i]
        ema20_val = ema20[i]
        vol_ok = volume_filter[i]
        bull_eng = bullish_engulf[i]
        bear_eng = bearish_engulf[i]
        
        if position == 0:
            # Long: price pulls back to EMA20 in uptrend (price > daily EMA50) with bullish engulfing
            if price > ema50 and price >= ema20_val * 0.995 and price <= ema20_val * 1.005 and vol_ok and bull_eng:
                signals[i] = 0.25
                position = 1
            # Short: price rallies to EMA20 in downtrend (price < daily EMA50) with bearish engulfing
            elif price < ema50 and price >= ema20_val * 0.995 and price <= ema20_val * 1.005 and vol_ok and bear_eng:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below EMA20 or trend turns down
            if price < ema20_val or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above EMA20 or trend turns up
            if price > ema20_val or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_4H_Trend_1D_Pullback_Scalper"
timeframe = "4h"
leverage = 1.0