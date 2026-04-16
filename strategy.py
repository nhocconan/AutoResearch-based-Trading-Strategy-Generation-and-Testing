#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly volume confirmation and daily trend filter.
# Long when price breaks above Camarilla R3 level AND volume > 1.5x 20-period weekly average AND daily close > daily EMA50 (uptrend).
# Short when price breaks below Camarilla S3 level AND volume > 1.5x 20-period weekly average AND daily close < daily EMA50 (downtrend).
# Exit when price crosses the Camarilla pivot point (midpoint) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Works in both bull and bear markets by requiring trend alignment via daily EMA50 and volume confirmation for breakout validity.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (from previous 6h bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = prev_high[0] = prev_low[0] = np.nan  # First bar has no previous
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    camarilla_r3 = pivot + (range_hl * 1.1 / 4)
    camarilla_s3 = pivot - (range_hl * 1.1 / 4)
    camarilla_pp = pivot  # Pivot point for exit
    
    # === Weekly Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    # === Daily Indicators: EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    daily_uptrend = close_1d > ema_50_1d_aligned  # Will be aligned inside loop via index
    
    # Calculate daily EMA50 aligned array properly
    # We need to align the daily EMA values to 6h bars
    ema_50_1d_aligned_full = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    daily_uptrend_aligned = close > ema_50_1d_aligned_full  # Compare 6h close to aligned daily EMA50
    
    # === 6h ATR for Stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pp[i]) or
            np.isnan(volume_spike[i]) or np.isnan(daily_uptrend_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_uptrend = daily_uptrend_aligned[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla pivot point
            if price < camarilla_pp[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla pivot point
            if price > camarilla_pp[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND volume spike AND daily uptrend
            if price > camarilla_r3[i] and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike AND daily downtrend
            elif price < camarilla_s3[i] and vol_spike and not is_uptrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_VolumeSpike_DailyEMA50_V1"
timeframe = "6h"
leverage = 1.0