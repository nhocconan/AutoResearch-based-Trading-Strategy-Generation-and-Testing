#!/usr/bin/env python3
"""
1d_1w_PriceAction_Strategy
Hypothesis: On daily timeframe, price rejection at weekly key levels (support/resistance) with volume confirmation and ADX trend filter provides high-probability reversal signals in both bull and bear markets. Uses weekly pivot points calculated from prior week's OHLC. Weekly context ensures alignment with major market structure, reducing false signals. Target: 15-25 trades/year per symbol.
"""

name = "1d_1w_PriceAction_Strategy"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly OHLC arrays
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to daily
    weekly_pivot_d = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_d = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_d = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly trend: 5-period EMA of weekly close
    weekly_ema5 = pd.Series(weekly_close).ewm(span=5, adjust=False, min_periods=5).mean().values
    weekly_uptrend = weekly_close > weekly_ema5
    weekly_downtrend = weekly_close < weekly_ema5
    
    # Align weekly trend to daily
    weekly_uptrend_d = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_d = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume spike detection: current volume > 1.5 * 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    # ADX(14) for trend strength - use only when ADX < 25 (ranging market) for mean reversion
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_abs = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr_abs).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Ranging market filter: ADX < 25
    ranging_market = adx < 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get aligned values
        pivot = weekly_pivot_d[i]
        r1 = weekly_r1_d[i]
        s1 = weekly_s1_d[i]
        uptrend = weekly_uptrend_d[i]
        downtrend = weekly_downtrend_d[i]
        ranging = ranging_market[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: Price at or below S1 in ranging market with volume spike
            if ranging and vol_spike and close[i] <= s1 * 1.005 and low[i] <= s1:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above R1 in ranging market with volume spike
            elif ranging and vol_spike and close[i] >= r1 * 0.995 and high[i] >= r1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or weekly trend turns down
            if close[i] >= pivot * 0.995 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or weekly trend turns up
            if close[i] <= pivot * 1.005 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals