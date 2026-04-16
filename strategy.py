#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 (1d) AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S4 (1d) AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Camarilla R4/S4 represent strong breakout levels, 1d ADX ensures we trade with the trend,
# volume spike confirms institutional participation. Designed to capture strong trending moves in both bull and bear markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivots (based on previous day) ===
    # Camarilla levels use previous day's range
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first values to NaN (no previous day)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # === 1d Indicators: ADX (14-period) for trend filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = np.nan
    down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for volume MA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r4 = R4_aligned[i]
        s4 = S4_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below R3 (taking profits) or ADX weakens (<20) or volume spike ends
            # Calculate R3 for exit: R3 = prev_close + 1.1*(prev_high - prev_low)/4
            # We'll use a simpler exit: price < R4 * 0.995 or ADX < 20
            if price < r4 * 0.995 or adx_val < 20 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above S3 (taking profits) or ADX weakens (<20) or volume spike ends
            if price > s4 * 1.005 or adx_val < 20 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R4 AND ADX > 25 (strong trend) AND volume spike
            if price > r4 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 AND ADX > 25 (strong trend) AND volume spike
            elif price < s4 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_CamarillaR4S4_Breakout_1dADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0