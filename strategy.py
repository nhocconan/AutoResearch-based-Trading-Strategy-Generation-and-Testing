#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above R4 with 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending up).
# Short when price breaks below S4 with 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending down).
# Uses discrete position size 0.25. Camarilla R4/S4 are strong breakout levels, volume confirms participation,
# ADX ensures we only trade in trending markets to avoid false breakouts in ranging conditions.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 60-120 trades over 4 years (15-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on prior 6h bar) ===
    # Calculate pivot from previous bar to avoid look-ahead
    pivot = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3.0
    range_val = np.roll(high, 1) - np.roll(low, 1)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX (14-period) for trend filter ===
    # True Range
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r4_level = r4[i]
        s4_level = s4[i]
        adx_1d = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below pivot (mean reversion) or volume spike ends
            if price < pivot[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above pivot (mean reversion) or volume spike ends
            if price > pivot[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R4 AND 1d volume spike AND 1d ADX > 25 (strong uptrend)
            if price > r4_level and vol_spike and adx_1d > 25:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 AND 1d volume spike AND 1d ADX > 25 (strong downtrend)
            elif price < s4_level and vol_spike and adx_1d > 25:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_CamarillaR4S4_Breakout_1dVolumeSpike_ADXFilter_V1"
timeframe = "6h"
leverage = 1.0