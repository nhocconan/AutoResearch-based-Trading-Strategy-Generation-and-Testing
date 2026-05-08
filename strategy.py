#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day trading strategy using weekly trend filter, daily pivot breakout with volume confirmation.
# Weekly trend determines direction, daily Camarilla R3/S3 breakout provides entry, volume spike confirms.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves.

name = "1d_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend direction
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Get daily data once for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d_prev) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3 (most significant)
    r3_1d = close_1d_prev + range_1d * 1.1 / 2
    s3_1d = close_1d_prev - range_1d * 1.1 / 2
    
    # Align daily pivot levels to daily timeframe (no alignment needed as already daily)
    r3_1d_aligned = r3_1d
    s3_1d_aligned = s3_1d
    
    # Volume spike detection: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Volatility filter: daily ATR-based range filter
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ratio = atr / close  # ATR as percentage of price
    vol_filter = (atr_ratio >= 0.01) & (atr_ratio <= 0.05)  # Between 1% and 5% daily volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for weekly and daily calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema21_1w_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        vol_spike_today = volume_spike[i]
        volatility_ok = vol_filter[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above weekly EMA, in volatility range
            if (close[i] > r3_val and vol_spike_today and 
                close[i] > ema_trend and volatility_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, below weekly EMA, in volatility range
            elif (close[i] < s3_val and vol_spike_today and 
                  close[i] < ema_trend and volatility_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below weekly EMA
            if (close[i] < s3_val or close[i] < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR above weekly EMA
            if (close[i] > r3_val or close[i] > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals