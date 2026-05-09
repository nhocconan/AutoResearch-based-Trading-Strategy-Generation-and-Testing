#!/usr/bin/env python3
# Hypothesis: 4h volume-weighted RSI with 12h EMA trend filter and ADX regime filter
# Long when: RSI(14) > 55, 12h EMA(50) rising, ADX(14) > 25 (trending market)
# Short when: RSI(14) < 45, 12h EMA(50) falling, ADX(14) > 25 (trending market)
# Exit when: RSI crosses back to 50 OR trend reverses (EMA direction change)
# Uses volume-weighted RSI for better signal quality in both bull and bear markets.
# Position size: 0.25 to manage risk. Target: 20-40 trades/year.

name = "4h_VolWeightedRSI_12hEMA_ADX_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume-weighted RSI calculation
    def vwma(close_arr, volume_arr, window):
        return np.convolve(close_arr * volume_arr, np.ones(window), 'same') / np.convolve(volume_arr, np.ones(window), 'same')
    
    # Calculate RSI components
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Smoothed volume-weighted RS
    alpha = 1.0 / 14
    avg_vg = np.zeros_like(vol_gain)
    avg_vl = np.zeros_like(vol_loss)
    
    avg_vg[0] = vol_gain[0]
    avg_vl[0] = vol_loss[0]
    
    for i in range(1, n):
        avg_vg[i] = alpha * vol_gain[i] + (1 - alpha) * avg_vg[i-1]
        avg_vl[i] = alpha * vol_loss[i] + (1 - alpha) * avg_vl[i-1]
    
    rs = avg_vg / (avg_vl + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close']
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_rising = ema_50_12h > ema_50_12h_prev
    ema_falling = ema_50_12h < ema_50_12h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    
    # ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = high[0] - low[0]
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(plus_dm)
        minus_di = np.zeros_like(minus_dm)
        
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        plus_di[period-1] = 100 * np.mean(plus_dm[:period]) / atr[period-1]
        minus_di[period-1] = 100 * np.mean(minus_dm[:period]) / atr[period-1]
        
        # Smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / period) / atr[i]
            minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / period) / atr[i]
        
        # DX and ADX
        dx = np.zeros_like(tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close)
    adx_trending = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i]) or np.isnan(adx_trending[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI > 55, 12h EMA rising, ADX > 25 (trending)
            if (vw_rsi[i] > 55 and 
                ema_rising_aligned[i] and 
                adx_trending[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI < 45, 12h EMA falling, ADX > 25 (trending)
            elif (vw_rsi[i] < 45 and 
                  ema_falling_aligned[i] and 
                  adx_trending[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses below 50 OR trend turns down
            if (vw_rsi[i] < 50) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 OR trend turns up
            if (vw_rsi[i] > 50) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals