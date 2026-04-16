#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h volume spike and 1d ADX regime filter
# Long when price > Camarilla R3 AND 12h volume > 1.8x 20-period volume SMA AND 1d ADX < 25 (range)
# Short when price < Camarilla S3 AND 12h volume > 1.8x 20-period volume SMA AND 1d ADX < 25 (range)
# Exit on price returning to Camarilla Pivot point or ATR stoploss (2.0 ATR)
# Uses discrete position sizing (0.25) to limit fee drag
# Camarilla levels provide precise intraday support/resistance; volume filter confirms breakout strength
# ADX < 25 ensures we only trade in ranging markets where mean reversion at pivot levels works
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for Camarilla levels and ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 1d Indicator: Camarilla levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's OHLC)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_pivot = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        camarilla_r3[i] = camarilla_pivot[i] + 1.1 * (prev_high - prev_low) / 4.0
        camarilla_s3[i] = camarilla_pivot[i] - 1.1 * (prev_high - prev_low) / 4.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # === 1d Indicator: ADX (14-period) for regime filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 28)  # 12h vol SMA, Camarilla levels, 1d ADX need ~30 bars
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_sma_20_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.8x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 1.8
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Regime filter: 1d ADX < 25 (ranging market)
        range_filter = adx_1d_aligned[i] < 25
        
        # Price levels
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to pivot or ATR stoploss
            if price <= pivot or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to pivot or ATR stoploss
            if price >= pivot or price >= entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > Camarilla R3 AND volume confirmation AND ranging market
            if price > r3 and vol_confirm and range_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < Camarilla S3 AND volume confirmation AND ranging market
            elif price < s3 and vol_confirm and range_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Camarilla_R3S3_12hVolSpike1.8x_1dADX_Range_v1"
timeframe = "4h"
leverage = 1.0