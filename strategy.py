#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 + 4h EMA34 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below 1h Camarilla S3 + 4h EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee drag and control drawdown.
# 4h EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-30 trades/year to minimize fee drag on 1h timeframe.

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA34 ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1h Camarilla Pivot Levels (based on previous day) ===
    # Calculate using previous day's high, low, close
    # Camarilla levels: R4 = close + ((high - low) * 1.1/2), R3 = close + ((high - low) * 1.1/4), etc.
    # We need to get the previous day's OHLC for each 1h bar
    
    # Create arrays to store previous day's OHLC aligned to 1h bars
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    # Get 1d data for previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # For each 1h bar, find the previous day's OHLC
    # We'll use the 1d data to get the previous completed day's values
    for i in range(n):
        current_time = prices['open_time'].iloc[i]
        # Find the 1d bar that completed before current_time
        # Get index of the 1d bar that is strictly before current_time
        mask = df_1d['open_time'] < current_time
        if mask.any():
            idx = mask.sum() - 1  # Get the last completed 1d bar
            if idx >= 0:
                prev_day_high[i] = df_1d['high'].iloc[idx]
                prev_day_low[i] = df_1d['low'].iloc[idx]
                prev_day_close[i] = df_1d['close'].iloc[idx]
    
    # Calculate Camarilla levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            high_low_diff = prev_day_high[i] - prev_day_low[i]
            camarilla_r3[i] = prev_day_close[i] + (high_low_diff * 1.1 / 4)
            camarilla_s3[i] = prev_day_close[i] - (high_low_diff * 1.1 / 4)
    
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
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above camarilla R3 (close > R3)
        # 2. 4h EMA34 uptrend (price > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3[i]) and \
           (close[i] > ema_34_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below camarilla S3 (close < S3)
        # 2. 4h EMA34 downtrend (price < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3[i]) and \
             (close[i] < ema_34_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA34_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0