#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above 6h Camarilla R1 + 1w EMA200 uptrend + volume > 2.0x 24-period avg
# Short when price breaks below 6h Camarilla S1 + 1w EMA200 downtrend + volume > 2.0x 24-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1w EMA200 provides strong long-term trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-30 trades/year to minimize fee drag on 6h timeframe.
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
    
    # Get 1w HTF data once before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === 1d Indicator: Previous day's OHLC for Camarilla levels ===
    # We need the previous completed day's high, low, close for each 6h bar
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    for i in range(n):
        current_time = prices['open_time'].iloc[i]
        # Find the 1d bar that completed before current_time
        mask = df_1d['open_time'] < current_time
        if mask.any():
            idx = mask.sum() - 1  # Get the last completed 1d bar
            if idx >= 0:
                prev_day_high[i] = df_1d['high'].iloc[idx]
                prev_day_low[i] = df_1d['low'].iloc[idx]
                prev_day_close[i] = df_1d['close'].iloc[idx]
    
    # Calculate Camarilla R1 and S1 levels (based on previous day)
    # Camarilla: R1 = close + ((high - low) * 1.1/12), S1 = close - ((high - low) * 1.1/12)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            high_low_diff = prev_day_high[i] - prev_day_low[i]
            camarilla_r1[i] = prev_day_close[i] + (high_low_diff * 1.1 / 12)
            camarilla_s1[i] = prev_day_close[i] - (high_low_diff * 1.1 / 12)
    
    # === 1w Indicator: EMA200 for trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume SMA for confirmation (using 24-period = 6 days of 6h bars)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 24) + 5  # EMA200 + volume(24) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above camarilla R1 (close > R1)
        # 2. 1w EMA200 uptrend (close > EMA200)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1[i]) and \
           (close[i] > ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below camarilla S1 (close < S1)
        # 2. 1w EMA200 downtrend (close < EMA200)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1[i]) and \
             (close[i] < ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R1S1_1wEMA200_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0