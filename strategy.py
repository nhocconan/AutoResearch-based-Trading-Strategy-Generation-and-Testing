#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w ADX trend filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 1.2x 20-period average AND 1w ADX > 20.
# Short when price breaks below Camarilla S3 AND 1d volume > 1.2x 20-period average AND 1w ADX > 20.
# Exit when price reaches Camarilla R4/S4 (profit target) or crosses R3/S3 in opposite direction (stop).
# Uses discrete position size 0.25. Designed to capture intraday momentum in trending markets.
# Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (from prior day) ===
    df_6h = get_htf_data(prices, '6h')
    # Camarilla uses prior daily OHLC, so we need 1d data for pivot calculation
    df_1d_prior = get_htf_data(prices, '1d')
    high_1d = df_1d_prior['high'].values
    low_1d = df_1d_prior['low'].values
    close_1d = df_1d_prior['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    hl_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * hl_range * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * hl_range * 1.1 / 4
    camarilla_r4 = close_1d + 1.1 * hl_range * 1.1 / 2
    camarilla_s4 = close_1d - 1.1 * hl_range * 1.1 / 2
    
    # Align Camarilla levels from 1d to 6h timeframe (prior day's levels apply to current 6h bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_prior, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_prior, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d_prior, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d_prior, camarilla_s4)
    
    # === 1d Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # === 1w Indicators: ADX > 20 (trending market filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = pd.Series(low_1w).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    strong_trend = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 6h ATR for dynamic stoploss (optional)
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = atr_6h_raw  # Already aligned as primary timeframe
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or
            np.isnan(atr_6h_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_6h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reaches R4 (profit target)
            if price >= camarilla_r4_aligned[i]:
                exit_signal = True
            # Exit if price crosses back below R3 (stop loss)
            elif price < camarilla_r3_aligned[i]:
                exit_signal = True
            # Optional: ATR-based trailing stop (2*ATR from high)
            # elif price < highest_since_entry - 2.0 * atr_val:
            #     exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reaches S4 (profit target)
            if price <= camarilla_s4_aligned[i]:
                exit_signal = True
            # Exit if price crosses back above S3 (stop loss)
            elif price > camarilla_s3_aligned[i]:
                exit_signal = True
            # Optional: ATR-based trailing stop (2*ATR from low)
            # elif price > lowest_since_entry + 2.0 * atr_val:
            #     exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND volume spike AND strong trending market
            if price > camarilla_r3_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike AND strong trending market
            elif price < camarilla_s3_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dVolumeSpike_1wADX_V1"
timeframe = "6h"
leverage = 1.0