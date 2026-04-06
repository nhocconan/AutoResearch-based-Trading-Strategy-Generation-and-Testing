#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adx_trend_4h_sma200"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX calculation for trend strength
    def calculate_adx(high, low, close, period=14):
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_sum = np.zeros(n)
        plus_dm_sum = np.zeros(n)
        minus_dm_sum = np.zeros(n)
        
        if len(tr) >= period:
            tr_sum[period] = np.sum(tr[:period])
            plus_dm_sum[period] = np.sum(plus_dm[:period])
            minus_dm_sum[period] = np.sum(minus_dm[:period])
            
            for i in range(period + 1, n):
                tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i-1]
                plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i-1]
                minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i-1]
        
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if tr_sum[i] != 0:
                plus_di[i] = 100 * (plus_dm_sum[i] / tr_sum[i])
                minus_di[i] = 100 * (minus_dm_sum[i] / tr_sum[i])
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.full(n, np.nan)
        if n >= 2 * period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    # SMA calculation
    def sma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        sma_val = np.full(n, np.nan)
        sma_val[period-1] = np.mean(arr[:period])
        for i in range(period, n):
            sma_val[i] = (sma_val[i-1] * (period-1) + arr[i]) / period
        return sma_val
    
    # ADX for trend detection
    adx = calculate_adx(high, low, close, 14)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    sma_200_4h = sma(close_4h, 200)
    sma_200_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_200_4h)
    
    # Get 1d data for longer-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = sma(close_1d, 50)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 100)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx[i]) or np.isnan(sma_200_4h_aligned[i]) or np.isnan(sma_50_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: ADX falls below 20 (trend weakening) or price closes below 1d SMA50
            if adx[i] < 20 or close[i] < sma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: ADX falls below 20 or price closes above 1d SMA50
            if adx[i] < 20 or close[i] > sma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries - only when ADX > 25 (strong trend)
            if adx[i] > 25:
                # Long: price above both 4h SMA200 and 1d SMA50
                if close[i] > sma_200_4h_aligned[i] and close[i] > sma_50_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: price below both 4h SMA200 and 1d SMA50
                elif close[i] < sma_200_4h_aligned[i] and close[i] < sma_50_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals