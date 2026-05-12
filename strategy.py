#!/usr/bin/env python3
name = "6h_ADX_ADXR_Trend_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 140:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # ADX and ADXR calculation on 6h data (period=14)
    period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first TR
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # ADXR = (ADX_t + ADX_{t-period}) / 2
    adxr = np.zeros(n)
    adxr[:period] = np.nan
    for i in range(period, n):
        adxr[i] = (adx[i] + adx[i-period]) / 2
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 140  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(adx[i]) or
            np.isnan(adxr[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > ADXR (trend strengthening) + DI+ > DI- + above daily EMA100 + volume filter
            if adx[i] > adxr[i] and plus_di[i] > minus_di[i] and close[i] > ema_100_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > ADXR (trend strengthening) + DI- > DI+ + below daily EMA100 + volume filter
            elif adx[i] > adxr[i] and minus_di[i] > plus_di[i] and close[i] < ema_100_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakening (ADX < ADXR) or DI- > DI+ or below daily EMA100
            if adx[i] < adxr[i] or minus_di[i] > plus_di[i] or close[i] < ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakening (ADX < ADXR) or DI+ > DI- or above daily EMA100
            if adx[i] < adxr[i] or plus_di[i] > minus_di[i] or close[i] > ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals