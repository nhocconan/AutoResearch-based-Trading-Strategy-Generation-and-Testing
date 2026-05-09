#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1d trend filter and 1w volatility filter.
# Long when price > VWAP(20) + 0.5*ATR(14) and 1d EMA(50) up and 1w ATR(14) < 1.5*1w ATR(50) (low volatility regime).
# Short when price < VWAP(20) - 0.5*ATR(14) and 1d EMA(50) down and 1w ATR(14) < 1.5*1w ATR(50).
# Uses VWAP as dynamic mean reversion target and volatility filter to avoid choppy markets.
# Designed to work in both bull and bear markets by following 1d EMA trend direction.
# VWAP deviation strategy avoids whipsaw in ranges and captures momentum in trends.
name = "6h_VWAPDeviation_1dTrend_1wVolFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP (20-period)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.zeros(n)
    vwap_den = np.zeros(n)
    for i in range(n):
        vwap_num[i] = typical_price[i] * volume[i]
        vwap_den[i] = volume[i]
        if i >= 20:
            vwap_num[i] = vwap_num[i-19:i+1].sum()
            vwap_den[i] = vwap_den[i-19:i+1].sum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.mean(tr[:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w ATR volatility filter (14 and 50 period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1w = high_1w - low_1w
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    trw[0] = tr1w[0]
    atr14_1w = np.zeros(len(df_1w))
    atr50_1w = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if i < 14:
            atr14_1w[i] = np.nan
        elif i == 14:
            atr14_1w[i] = np.mean(trw[:15])
        else:
            atr14_1w[i] = (atr14_1w[i-1] * 13 + trw[i]) / 14
        if i < 50:
            atr50_1w[i] = np.nan
        elif i == 50:
            atr50_1w[i] = np.mean(trw[:51])
        else:
            atr50_1w[i] = (atr50_1w[i-1] * 49 + trw[i]) / 50
    vol_filter = atr14_1w < (1.5 * atr50_1w)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1w, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_dev = 0.5 * atr[i]
        
        if position == 0:
            # Long: price > VWAP + 0.5*ATR + 1d EMA up + low vol regime
            if (price > vwap[i] + vwap_dev and ema_1d_aligned[i] > ema_1d_aligned[i-1] and vol_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < VWAP - 0.5*ATR + 1d EMA down + low vol regime
            elif (price < vwap[i] - vwap_dev and ema_1d_aligned[i] < ema_1d_aligned[i-1] and vol_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP or 1d EMA turns down
            if price < vwap[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP or 1d EMA turns up
            if price > vwap[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals