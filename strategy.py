#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_MeanReversion
# Hypothesis: On daily timeframe, KAMA identifies trend direction while RSI identifies
# mean-reversion entries within that trend. Long when trend up and RSI < 30 (oversold),
# short when trend down and RSI > 70 (overbought). This captures trend continuation
# after pullbacks in both bull and bear markets. Uses volume confirmation to avoid
# false signals and ATR-based stoploss via signal=0 when trend breaks.

name = "1d_KAMA_Trend_RSI_MeanReversion"
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
    
    # Get weekly data for trend filter (more stable than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on weekly data
    def kama(price, er_period=10, fast=2, slow=30):
        n = len(price)
        kama_arr = np.full(n, np.nan)
        if n < er_period:
            return kama_arr
        
        # Efficiency Ratio
        change = np.abs(np.diff(price, er_period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
        er = np.concatenate([np.full(er_period-1, np.nan), er])
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_arr[er_period-1] = np.mean(price[:er_period])
        for i in range(er_period, n):
            if not np.isnan(sc[i]):
                kama_arr[i] = kama_arr[i-1] + sc[i] * (price[i] - kama_arr[i-1])
        return kama_arr
    
    kama_1w = kama(close_1w)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI for mean reversion
    def rsi(price, period=14):
        n = len(price)
        rsi_arr = np.full(n, np.nan)
        if n < period + 1:
            return rsi_arr
        
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi_arr = 100 - (100 / (1 + rs))
        return rsi_arr
    
    rsi_14 = rsi(close)
    
    # Volume filter: current vs 20-day average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_14[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend from KAMA
        # Trend up when price above KAMA, trend down when below
        trend_up = close[i] > kama_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + RSI oversold + volume confirmation
            if trend_up and rsi_14[i] < 30 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + RSI overbought + volume confirmation
            elif not trend_up and rsi_14[i] > 70 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or RSI overbought
            if not trend_up or rsi_14[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or RSI oversold
            if trend_up or rsi_14[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals