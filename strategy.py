#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 6h Camarilla R3 + 1w EMA50 uptrend + volume > 1.6x 20-period avg
# Short when price breaks below 6h Camarilla S3 + 1w EMA50 downtrend + volume > 1.6x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1w EMA50 provides strong multi-week trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.6x) targets ~15-30 trades/year to minimize fee drag on 6h timeframe.
# Camarilla levels calculated from previous day's OHLC using 1d HTF data.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Previous day's OHLC for Camarilla levels ===
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    # For each 6h bar, find the previous day's OHLC from 1d data
    for i in range(n):
        current_time = prices['open_time'].iloc[i]
        # Get index of the last completed 1d bar before current_time
        mask = df_1d['open_time'] < current_time
        if mask.any():
            idx = mask.sum() - 1  # Last completed 1d bar
            if idx >= 0:
                prev_day_high[i] = df_1d['high'].iloc[idx]
                prev_day_low[i] = df_1d['low'].iloc[idx]
                prev_day_close[i] = df_1d['close'].iloc[idx]
    
    # Calculate Camarilla levels (R3 and S3)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            high_low_diff = prev_day_high[i] - prev_day_low[i]
            camarilla_r3[i] = prev_day_close[i] + (high_low_diff * 1.1 / 4)
            camarilla_s3[i] = prev_day_close[i] - (high_low_diff * 1.1 / 4)
    
    # === 1w Indicator: EMA50 ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50(1w) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.6x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.6)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above camarilla R3 (close > R3)
        # 2. 1w EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3[i]) and \
           (close[i] > ema_50_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below camarilla S3 (close < S3)
        # 2. 1w EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3[i]) and \
             (close[i] < ema_50_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_1wEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0