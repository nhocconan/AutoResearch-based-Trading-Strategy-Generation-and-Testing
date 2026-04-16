#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w ADX trend filter.
# Long when price breaks above Camarilla R3 level AND 1d volume > 1.5x 20-period average AND 1w ADX > 25.
# Short when price breaks below Camarilla S3 level AND 1d volume > 1.5x 20-period average AND 1w ADX > 25.
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite Camarilla break (R3/S3).
# Uses discrete position size 0.28. Works in both bull and bear markets by requiring
# volume confirmation, ADX trend filter, and using symmetric pivot levels.
# Target: 75-150 total trades over 4 years (19-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous 12h bar) ===
    # Calculate pivots using previous bar's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4.0)  # R3 = pivot + (range * 1.1/4)
    s3 = pivot - (range_val * 1.1 / 4.0)  # S3 = pivot - (range * 1.1/4)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: ADX for trend filter (ADX > 25) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = pd.Series(low_1w).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed values (using Wilder's smoothing, alpha=1/period)
    period_adx = 14
    atr_1w_raw = pd.Series(tr_1w).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_1w_raw)
    di_minus = 100 * (dm_minus_smooth / atr_1w_raw)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1w_raw = pd.Series(dx).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx*2).mean().values
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w_raw)
    adx_filter = adx_1w_aligned > 25.0
    
    # === 12h ATR for stoploss ===
    tr1_12h = pd.Series(high).diff()
    tr2_12h = pd.Series(low).diff().abs()
    tr3_12h = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 14*2 + 20 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(volume_spike[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(atr_12h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_ok = adx_1w_aligned[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3 (opposite breakout)
            if price < s3[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3 (opposite breakout)
            if price > r3[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND volume spike AND ADX > 25
            if price > r3[i] and vol_spike and adx_ok:
                signals[i] = 0.28
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike AND ADX > 25
            elif price < s3[i] and vol_spike and adx_ok:
                signals[i] = -0.28
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dVolume_1wADXFilter_V1"
timeframe = "12h"
leverage = 1.0