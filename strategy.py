#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (HMA21) and volume confirmation.
# Long when price breaks above Camarilla R3 AND 4h HMA trending up AND volume > 1.3x 20-period average.
# Short when price breaks below Camarilla S3 AND 4h HMA trending down AND volume > 1.3x 20-period average.
# Exit on opposite Camarilla level (S3 for long, R3 for short).
# Uses discrete position size 0.20. Target: 60-150 trades over 4 years (15-37/year) to minimize fee drag.
# Session filter (08-20 UTC) reduces noise. Works in both bull/bear by requiring 4h trend and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla levels (based on previous day) ===
    # Camarilla uses previous day's high, low, close
    # We approximate using rolling window of 24 periods (24*1h = 1 day)
    lookback = 24
    if n < lookback:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1)
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1)
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1)
    
    # Avoid division by zero
    range_val = prev_high - prev_low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    camarilla_r3 = prev_close + (range_val * 1.1 / 4)
    camarilla_s3 = prev_close - (range_val * 1.1 / 4)
    camarilla_r4 = prev_close + (range_val * 1.1 / 2)
    camarilla_s4 = prev_close - (range_val * 1.1 / 2)
    
    # === 4h Indicators: HMA(21) for trend ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 10  # 21/2 rounded
    sqrt_len = 4   # sqrt(21) rounded
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_up = hma_4h_aligned > np.roll(hma_4h_aligned, 1)
    hma_down = hma_4h_aligned < np.roll(hma_4h_aligned, 1)
    
    # === 4h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.3 * vol_ma_4h_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3
            if price < camarilla_s3[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3
            if price > camarilla_r3[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND 4h HMA trending up AND volume spike
            if price > camarilla_r3[i] and hma_up[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND 4h HMA trending down AND volume spike
            elif price < camarilla_s3[i] and hma_down[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_CamarillaR3S3_4hHMA21_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0