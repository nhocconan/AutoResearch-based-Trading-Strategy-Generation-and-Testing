#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) identifies trend direction with reduced lag.
RSI (14) filters overbought/oversold conditions, and Choppiness Index (14) identifies ranging vs trending markets.
Long when KAMA rising, RSI < 50, and CHOP > 61.8 (ranging market - mean reversion opportunity).
Short when KAMA falling, RSI > 50, and CHOP > 61.8.
Uses 1d ADX as trend strength filter to avoid whipsaws in weak trends.
Designed for 4h timeframe with 20-40 trades/year to minimize fee drag.
Works in both bull and bear markets by adapting to ranging conditions.
"""

name = "4h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "4h"
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
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def calculate_kama(close, slow=2, fast=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < 10:
            return kama
        
        # Change and volatility
        change = np.abs(np.diff(close, k=10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
        
        # Avoid division by zero
        volatility = np.where(volatility == 0, 1e-10, volatility)
        
        # Efficiency Ratio
        er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
        er = np.concatenate([np.full(9, np.nan), er])  # Align with close index
        
        # Smoothing Constants
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        
        # KAMA calculation
        kama[9] = close[9]  # Start with first close
        for i in range(10, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    kama = calculate_kama(close, slow=2, fast=30)
    
    # RSI (14) on 4h close
    def calculate_rsi(close, period=14):
        n = len(close)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        # RS and RSI
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Choppiness Index (14) on 4h data
    def calculate_chop(high, low, close, period=14):
        n = len(close)
        chop = np.full(n, np.nan)
        if n < period:
            return chop
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with index
        
        # Sum of True Range over period
        tr_sum = np.full(n, np.nan)
        for i in range(period, n):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.full(n, np.nan)
        min_low = np.full(n, np.nan)
        for i in range(period-1, n):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        for i in range(period-1, n):
            if not np.isnan(tr_sum[i]) and max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0  # Neutral when range is zero
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    
    # ADX (14) on 1d for trend strength filter
    def calculate_adx(high, low, close, period=14):
        n = len(close)
        adx = np.full(n, np.nan)
        if n < period * 2:
            return adx
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        plus_dm_smooth = np.full(n, np.nan)
        minus_dm_smooth = np.full(n, np.nan)
        
        # Initial averages
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX (smoothed DX)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: rising or falling
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi[i] < 50
        rsi_not_oversold = rsi[i] > 50
        
        # Chop filter: ranging market (mean reversion opportunity)
        chop_high = chop[i] > 61.8  # Ranging market
        
        # ADX filter: avoid weak trends
        strong_trend = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: KAMA rising, RSI < 50, ranging market, strong trend
            if kama_rising and rsi_not_overbought and chop_high and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI > 50, ranging market, strong trend
            elif kama_falling and rsi_not_oversold and chop_high and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA falling or chop low (trending market)
            if not kama_rising or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rising or chop low (trending market)
            if not kama_falling or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals