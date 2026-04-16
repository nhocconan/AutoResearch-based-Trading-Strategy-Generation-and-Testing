#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d ADX trend filter.
# Long when price breaks above 1h Camarilla R3 level AND 4h volume > 1.5x 20-period average AND 1d ADX > 20 (trending market).
# Short when price breaks below 1h Camarilla S3 level AND 4h volume > 1.5x 20-period average AND 1d ADX > 20.
# Exit when price crosses the 1h Camarilla midpoint (PP) or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.20. Session filter: 08-20 UTC.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by requiring trend (ADX>20) and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Calculate from previous completed bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 4.0)
    s3 = pivot - (range_val * 1.1 / 4.0)
    pp = pivot  # Camarilla pivot point (midpoint)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.5 * vol_ma_4h_aligned)
    
    # === 1d Indicators: ADX > 20 (trending market filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 1h ATR for stoploss
    tr1_1h = pd.Series(high).diff()
    tr2_1h = pd.Series(low).diff().abs()
    tr3_1h = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1_1h, tr2_1h, tr3_1h], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(pp[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or np.isnan(atr_1h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_1h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla pivot point (PP)
            if price < pp[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla pivot point (PP)
            if price > pp[i]:
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
            # LONG: Price breaks above Camarilla R3 AND volume spike AND trending market
            if price > r3[i] and vol_spike and is_strong_trend:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike AND trending market
            elif price < s3[i] and vol_spike and is_strong_trend:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_CamarillaR3S3_4hVolumeSpike_1dADX_V1"
timeframe = "1h"
leverage = 1.0