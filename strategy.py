#!/usr/bin/env python3
# 6h_1d_cci_adx_v1
# Hypothesis: Combines CCI (20) for momentum extremes and ADX (14) for trend strength on 6h timeframe.
# Uses 1d ADX to filter regime: only trade when 1d ADX > 25 (trending market).
# Long when 6h CCI crosses above -100 from below AND 6h ADX > 25.
# Short when 6h CCI crosses below 100 from above AND 6h ADX > 25.
# Exit when CCI returns to neutral zone (-100 to 100).
# Designed to capture strong trends while avoiding choppy markets.
# Target: 15-30 trades/year (60-120 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX(14) for 6h trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])),
                                              np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
        # Directional Movement
        up_move = high - np.concatenate([[high[0]], high[:-1]])
        down_move = np.concatenate([[low[0]], low[:-1]]) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr_sum = np.zeros(n)
        plus_dm_sum = np.zeros(n)
        minus_dm_sum = np.zeros(n)
        
        tr_sum[period-1] = np.sum(tr[:period])
        plus_dm_sum[period-1] = np.sum(plus_dm[:period])
        minus_dm_sum[period-1] = np.sum(minus_dm[:period])
        
        for i in range(period, n):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
        
        # Directional Indicators
        plus_di = np.where(tr_sum > 0, 100 * plus_dm_sum / tr_sum, 0)
        minus_di = np.where(tr_sum > 0, 100 * minus_dm_sum / tr_sum, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = np.zeros(n)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if 2*period-1 <= n else 0
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    
    # Calculate CCI(20)
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = np.zeros(n)
        for i in range(period-1, n):
            sma_tp[i] = np.mean(tp[i-period+1:i+1])
        
        mad = np.zeros(n)
        for i in range(period-1, n):
            mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - sma_tp[i]))
        
        cci = np.zeros(n)
        cci[period-1:] = (tp[period-1:] - sma_tp[period-1:]) / (0.015 * mad[period-1:])
        return cci
    
    cci_6h = calculate_cci(high, low, close, 20)
    
    # Load 1d data ONCE for regime filter (ADX > 25 = trending)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(adx_6h[i]) or np.isnan(cci_6h[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        regime_filter = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: CCI returns below 100 (return to neutral)
            if cci_6h[i] < 100 and cci_6h[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns above -100 (return to neutral)
            if cci_6h[i] > -100 and cci_6h[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI crosses above -100 from below AND ADX > 25
            if cci_6h[i] > -100 and cci_6h[i-1] <= -100 and regime_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI crosses below 100 from above AND ADX > 25
            elif cci_6h[i] < 100 and cci_6h[i-1] >= 100 and regime_filter:
                position = -1
                signals[i] = -0.25
    
    return signals