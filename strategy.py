#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1w trend filter.
# Uses 1d Camarilla levels (R3, S3, R4, S4) calculated from prior 1d bar.
# Long when close breaks above R4 with volume > 2.0x 20-period 1d average AND 1w close > 1w EMA20 (uptrend).
# Short when close breaks below S4 with volume > 2.0x 20-period 1d average AND 1w close < 1w EMA20 (downtrend).
# Exit when price reverts to R3/S3 level or ATR-based stop (1.5*ATR).
# Uses discrete position size 0.25. Designed to capture strong breakouts in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring 1w trend filter and volume spike, avoiding false breakouts in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Camarilla levels (from prior 1d bar) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current 6h bar using prior 1d bar (shifted by 1)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    #          R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    rng_1d = high_1d - low_1d
    camarilla_r4 = close_1d + (1.1 * rng_1d * 1.1 / 2)
    camarilla_s4 = close_1d - (1.1 * rng_1d * 1.1 / 2)
    camarilla_r3 = close_1d + (1.1 * rng_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (1.1 * rng_1d * 1.1 / 4)
    
    # Align to 6h timeframe (prior 1d bar's levels are available at 6h open)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1w Indicators: Trend filter (close > EMA20 for uptrend, close < EMA20 for downtrend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    uptrend_1w = close_1w > ema_20_1w_aligned  # using aligned array for comparison
    downtrend_1w = close_1w < ema_20_1w_aligned
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 6h ATR for stoploss
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(uptrend_1w[i]) or np.isnan(downtrend_1w[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_uptrend = uptrend_1w[i]
        is_downtrend = downtrend_1w[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to R3 level
            if price <= camarilla_r3_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to S3 level
            if price >= camarilla_s3_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Close breaks above R4 with volume spike AND 1w uptrend
            if close[i] > camarilla_r4_aligned[i] and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Close breaks below S4 with volume spike AND 1w downtrend
            elif close[i] < camarilla_s4_aligned[i] and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dVolumeSpike_1wTrend_V1"
timeframe = "6h"
leverage = 1.0