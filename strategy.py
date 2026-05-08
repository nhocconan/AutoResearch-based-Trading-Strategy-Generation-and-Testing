#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h ADX trend filter and volume confirmation.
# RSI(14) < 30 for long, > 70 for short in the direction of 4h ADX(14) > 25 trend.
# Volume > 1.5x 20-period average on 4h confirms participation.
# Uses 4h for signal direction (ADX trend), 1h only for entry timing.
# Session filter (08-20 UTC) reduces noise. Target: 15-30 trades/year.

name = "1h_RSI_MeanReversion_4hADX25_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX trend filter and volume average
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ADX(14) on 4h data
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            dm_plus[i] = max(high[i] - high[i-1], 0)
            dm_minus[i] = max(low[i-1] - low[i], 0)
            if dm_plus[i] == dm_minus[i]:
                dm_plus[i] = 0
                dm_minus[i] = 0
        
        atr = np.zeros(n)
        if n >= period:
            atr[period-1] = np.mean(tr[1:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        di_plus = np.zeros(n)
        di_minus = np.zeros(n)
        if n >= period and atr[period-1] != 0:
            dm_plus_smooth = np.zeros(n)
            dm_minus_smooth = np.zeros(n)
            dm_plus_smooth[period-1] = np.mean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.mean(dm_minus[1:period])
            for i in range(period, n):
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
            
            di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
            di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        dx = np.zeros(n)
        if n >= period:
            di_sum = di_plus + di_minus
            dx = np.where(di_sum != 0, 100 * np.abs(di_plus - di_minus) / di_sum, 0)
        
        adx = np.full(n, np.nan)
        if n >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Calculate 20-period average volume on 4h
    vol_avg_20_4h = np.full(len(volume_4h), np.nan)
    for i in range(20, len(volume_4h)):
        vol_avg_20_4h[i] = np.mean(volume_4h[i-20:i])
    
    # Calculate RSI(14) on 1h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Align all indicators to 1h timeframe
    adx_14_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_14_4h)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(adx_14_4h_aligned[i]) or np.isnan(vol_avg_20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_4h_aligned[i]):
            # Find current 4h bar's volume
            idx_4h = 0
            while idx_4h < len(df_4h) and df_4h.iloc[idx_4h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_4h += 1
            idx_4h -= 1  # last completed 4h bar
            
            if idx_4h >= 0:
                vol_4h_current = df_4h.iloc[idx_4h]['volume']
                vol_filter = vol_4h_current > 1.5 * vol_avg_20_4h_aligned[i]
        
        if position == 0:
            # Look for entry: RSI extreme + ADX trend (>25) + volume
            # Long when RSI < 30 in uptrend (ADX > 25)
            long_condition = (rsi[i] < 30) and (adx_14_4h_aligned[i] > 25) and vol_filter
            # Short when RSI > 70 in downtrend (ADX > 25)
            short_condition = (rsi[i] > 70) and (adx_14_4h_aligned[i] > 25) and vol_filter
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 or ADX < 20 (trend weakening)
            if (rsi[i] > 50) or (adx_14_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 or ADX < 20 (trend weakening)
            if (rsi[i] < 50) or (adx_14_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals