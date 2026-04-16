#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout (R1/S1) with 1w ADX(14) trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Exit when price crosses Camarilla pivot point (PP) OR volume drops below average.
# Uses discrete position size 0.25. Designed to capture strong trends in both bull and bear markets.
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on prior 12h bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    # We use prior bar's values to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no prior
    
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = PP + (prev_high - prev_low) * 1.1 / 12.0
    S1 = PP - (prev_high - prev_low) * 1.1 / 12.0
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = -pd.Series(low_1w).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 1w ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for volume MA, 1 for Camarilla)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(PP[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla PP OR volume spike ends
            if price < PP[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla PP OR volume spike ends
            if price > PP[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND 1w ADX > 25 (trend) AND volume spike
            if price > R1[i] and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND 1w ADX > 25 (trend) AND volume spike
            elif price < S1[i] and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_1wADX25_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0