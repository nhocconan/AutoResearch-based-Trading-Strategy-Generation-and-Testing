#!/usr/bin/env python3
# 1h_4h1d_Momentum_Reversal_Filter
# Hypothesis: Mean reversion on 1h with momentum and regime filters from 4h and 1d.
# In trending regimes (4h ADX > 25), trade pullbacks to EMA21 in trend direction.
# In ranging regimes (4h ADX < 20), fade extremes using 1h RSI with 1d trend filter.
# Volume confirmation required on all entries. Designed for 15-35 trades/year on 1h.

name = "1h_4h1d_Momentum_Reversal_Filter"
timeframe = "1h"
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
    
    # Get 4h data for ADX and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilder_smooth(x, period):
            result = np.full_like(x, np.nan)
            if len(x) < period:
                return result
            result[period-1] = np.nansum(x[1:period+1])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
            return result
        
        plus_di = 100 * wilder_smooth(plus_dm, period) / wilder_smooth(tr, period)
        minus_di = 100 * wilder_smooth(minus_dm, period) / wilder_smooth(tr, period)
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 4h EMA21 for trend direction
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        if len(close) >= period:
            avg_gain[period-1] = np.mean(gain[1:period+1])
            avg_loss[period-1] = np.mean(loss[1:period+1])
            for i in range(period, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close, 14)
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(adx_4h_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_1h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume_filter[i] if i < len(volume_filter) else False
        
        if position == 0:
            # Trending regime: ADX > 25
            if adx_4h_aligned[i] > 25:
                # Uptrend: price > EMA21_4h
                if close[i] > ema21_4h_aligned[i]:
                    # Long on pullback to EMA21 with RSI < 40
                    if close[i] <= ema21_4h_aligned[i] * 1.005 and rsi_1h[i] < 40 and vol_ok:
                        signals[i] = 0.20
                        position = 1
                # Downtrend: price < EMA21_4h
                elif close[i] < ema21_4h_aligned[i]:
                    # Short on pullback to EMA21 with RSI > 60
                    if close[i] >= ema21_4h_aligned[i] * 0.995 and rsi_1h[i] > 60 and vol_ok:
                        signals[i] = -0.20
                        position = -1
            # Ranging regime: ADX < 20
            elif adx_4h_aligned[i] < 20:
                # Use 1d trend filter: only trade in direction of 1d EMA50
                if close[i] > ema50_1d_aligned[i]:  # 1d uptrend
                    # Long when RSI oversold
                    if rsi_1h[i] < 30 and vol_ok:
                        signals[i] = 0.20
                        position = 1
                else:  # 1d downtrend
                    # Short when RSI overbought
                    if rsi_1h[i] > 70 and vol_ok:
                        signals[i] = -0.20
                        position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi_1h[i] > 70 or (adx_4h_aligned[i] > 25 and close[i] < ema21_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi_1h[i] < 30 or (adx_4h_aligned[i] > 25 and close[i] > ema21_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals