#!/usr/bin/env python3
"""
1h_4h1d_Trend_Momentum_v1
Hypothesis: Use 4h ADX for trend strength and 1d RSI for momentum, with 1h price action for entry timing.
Go long when 4h ADX > 25 (trending) AND 1d RSI > 55 AND price > 1h VWAP, short when 4h ADX > 25 AND 1d RSI < 45 AND price < 1h VWAP.
Uses session filter (08-20 UTC) to avoid low-liquidity hours. Target: 15-30 trades/year by requiring strong trend + momentum alignment.
Works in bull via long signals and bear via short signals when ADX confirms trend.
"""

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
    
    # Get 4h data for ADX
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h ADX(14)
    adx_period = 14
    adx_4h = np.full_like(close_4h, np.nan)
    
    if len(high_4h) >= adx_period * 2:
        # True Range
        tr1 = high_4h[1:] - low_4h[1:]
        tr2 = np.abs(high_4h[1:] - close_4h[:-1])
        tr3 = np.abs(low_4h[1:] - close_4h[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        plus_dm = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                           np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
        minus_dm = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                            np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        # Initial average
        if len(tr) >= adx_period + 1:
            atr[adx_period] = np.nanmean(tr[1:adx_period+1])
            plus_dm_smooth[adx_period] = np.nanmean(plus_dm[1:adx_period+1])
            minus_dm_smooth[adx_period] = np.nanmean(minus_dm[1:adx_period+1])
            
            # Wilder smoothing
            for i in range(adx_period + 1, len(tr)):
                atr[i] = (atr[i-1] * (adx_period - 1) + tr[i]) / adx_period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (adx_period - 1) + plus_dm[i]) / adx_period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (adx_period - 1) + minus_dm[i]) / adx_period
        
        # Directional Indicators
        plus_di = np.full_like(atr, np.nan)
        minus_di = np.full_like(atr, np.nan)
        dx = np.full_like(atr, np.nan)
        
        valid = ~np.isnan(atr) & (atr != 0)
        plus_di[valid] = 100 * plus_dm_smooth[valid] / atr[valid]
        minus_di[valid] = 100 * minus_dm_smooth[valid] / atr[valid]
        dx[valid] = 100 * np.abs(plus_di[valid] - minus_di[valid]) / (plus_di[valid] + minus_di[valid])
        
        # ADX: smoothed DX
        adx_4h = np.full_like(dx, np.nan)
        if len(dx) >= adx_period * 2:
            adx_4h[2*adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
            for i in range(2*adx_period, len(dx)):
                if not np.isnan(dx[i]):
                    adx_4h[i] = (adx_4h[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align 4h ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    rsi_period = 14
    rsi_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.nanmean(gain[:rsi_period])
        avg_loss[rsi_period] = np.nanmean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h VWAP (session-based reset)
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(adx_period*2, rsi_period+1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if position == 0 and in_session:
            # Long: ADX > 25 (trending) AND RSI > 55 AND price > VWAP
            if adx_4h_aligned[i] > 25 and rsi_1d_aligned[i] > 55 and close[i] > vwap[i]:
                signals[i] = 0.20
                position = 1
            # Short: ADX > 25 (trending) AND RSI < 45 AND price < VWAP
            elif adx_4h_aligned[i] > 25 and rsi_1d_aligned[i] < 45 and close[i] < vwap[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: ADX < 20 (no trend) OR RSI < 40
            if adx_4h_aligned[i] < 20 or rsi_1d_aligned[i] < 40:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: ADX < 20 (no trend) OR RSI > 60
            if adx_4h_aligned[i] < 20 or rsi_1d_aligned[i] > 60:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Trend_Momentum_v1"
timeframe = "1h"
leverage = 1.0