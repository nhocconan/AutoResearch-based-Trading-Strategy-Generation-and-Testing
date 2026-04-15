#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R4 + 12h EMA34 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below 4h Camarilla S4 + 12h EMA34 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 12h EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~20-40 trades/year to minimize fee drag on 4h timeframe.
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
    
    # === 1d OHLC for Camarilla calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Create arrays to store previous day's OHLC aligned to 4h bars
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    # For each 4h bar, find the previous day's OHLC
    for i in range(n):
        current_time = prices['open_time'].iloc[i]
        # Find the 1d bar that completed before current_time
        mask = df_1d['open_time'] < current_time
        if mask.any():
            idx = mask.sum() - 1  # Get the last completed 1d bar
            if idx >= 0:
                prev_day_high[i] = high_1d[idx]
                prev_day_low[i] = low_1d[idx]
                prev_day_close[i] = close_1d[idx]
    
    # Calculate Camarilla levels (R4 and S4)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            high_low_diff = prev_day_high[i] - prev_day_low[i]
            camarilla_r4[i] = prev_day_close[i] + (high_low_diff * 1.1 / 2)  # R4 = close + 1.1*(H-L)/2
            camarilla_s4[i] = prev_day_close[i] - (high_low_diff * 1.1 / 2)  # S4 = close - 1.1*(H-L)/2
    
    # Get 12h HTF data once before loop for EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: EMA34 ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above camarilla R4 (close > R4)
        # 2. 12h EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4[i]) and \
           (close[i] > ema_34_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below camarilla S4 (close < S4)
        # 2. 12h EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4[i]) and \
             (close[i] < ema_34_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R4S4_12hEMA34_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0