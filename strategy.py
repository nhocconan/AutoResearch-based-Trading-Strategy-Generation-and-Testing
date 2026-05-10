#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: KAMA trend direction on 1d with RSI(2) mean reversion and Choppiness Index regime filter.
# KAMA adapts to market noise, RSI(2) catches short-term reversals, Choppiness Index filters ranging vs trending.
# Works in bull markets via trend following, in bear markets via mean reversion in ranging conditions.
# Target: 20-70 trades over 4 years (5-18/year) to minimize fee drag.

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - 1d
    def calculate_kama(price, er_length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, prepend=price[0]))
        dir = np.abs(np.diff(price, prepend=price[0]))  # temporary
        for i in range(1, len(price)):
            dir[i] = np.abs(price[i] - price[i-er_length]) if i >= er_length else 0
        vol = np.zeros_like(price)
        for i in range(1, len(price)):
            vol[i] = vol[i-1] + np.abs(price[i] - price[i-1])
        
        er = np.zeros_like(price)
        er[:er_length] = 0
        for i in range(er_length, len(price)):
            if vol[i] != 0:
                er[i] = dir[i] / vol[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close)
    
    # RSI(2) for mean reversion signals
    def calculate_rsi(price, period=2):
        delta = np.diff(price, prepend=price[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 2)
    
    # Choppiness Index (14) - 1d
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if i == 1:
                atr[i] = tr
            else:
                atr[i] = (atr[i-1] * (period-1) + tr) / period
        
        sum_atr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # 1w EMA34 for trend filter (needs extra delay as it's trend-following)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align 1d indicators
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Choppiness Index
        # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long conditions
            if is_ranging and rsi_aligned[i] < 20:  # Oversold in ranging market
                signals[i] = 0.25
                position = 1
            elif is_trending and close[i] > kama_aligned[i] and close[i] > ema_34_1w_aligned[i]:  # Uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions
            elif is_ranging and rsi_aligned[i] > 80:  # Overbought in ranging market
                signals[i] = -0.25
                position = -1
            elif is_trending and close[i] < kama_aligned[i] and close[i] < ema_34_1w_aligned[i]:  # Downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit conditions
            if is_ranging and rsi_aligned[i] > 60:  # RSI mean reversion exit
                signals[i] = 0.0
                position = 0
            elif is_trending and close[i] < kama_aligned[i]:  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_ranging and rsi_aligned[i] < 40:  # RSI mean reversion exit
                signals[i] = 0.0
                position = 0
            elif is_trending and close[i] > kama_aligned[i]:  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals