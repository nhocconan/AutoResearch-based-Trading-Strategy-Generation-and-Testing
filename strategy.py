#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h ADX trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND 12h ADX > 25 AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 AND 12h ADX > 25 AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or ATR(14) stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to capture institutional breakouts with volume and trend confirmation.
# Works in both bull and bear markets by requiring strong trend (ADX>25) and volume confirmation, avoiding false breakouts in chop.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from previous day's OHLC
    # We use rolling window of 24 (4h bars in 1d) to get previous day's OHLC
    # But since we need previous day's close, we shift by 1
    roll_high_24 = pd.Series(high).rolling(window=24, min_periods=24).max()
    roll_low_24 = pd.Series(low).rolling(window=24, min_periods=24).min()
    roll_close_24 = pd.Series(close).rolling(window=24, min_periods=24).last()
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = roll_high_24.shift(1).values
    prev_low = roll_low_24.shift(1).values
    prev_close = roll_close_24.shift(1).values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    
    # === 12h Indicators: ADX(14) for trend strength ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_12h).diff()
    dm_minus = pd.Series(low_12h).diff().abs()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed DM and TR
    si = pd.Series(index=range(len(dm_plus)))
    si[:] = np.nan
    if len(dm_plus) >= 14:
        si[13] = np.nansum(dm_plus[:14])
        si[14:] = si[13:-1] - (si[13:-1] / 14) + dm_plus[14:]
    dm_plus_smooth = si.values
    
    si = pd.Series(index=range(len(dm_minus)))
    si[:] = np.nan
    if len(dm_minus) >= 14:
        si[13] = np.nansum(dm_minus[:14])
        si[14:] = si[13:-1] - (si[13:-1] / 14) + dm_minus[14:]
    dm_minus_smooth = si.values
    
    si = pd.Series(index=range(len(atr_12h)))
    si[:] = np.nan
    if len(atr_12h) >= 14:
        si[13] = np.nansum(atr_12h[:14])
        si[14:] = si[13:-1] - (si[13:-1] / 14) + atr_12h[14:]
    tr_smooth = si.values
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_strong = adx > 25
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_strong)
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = vol_12h > (1.5 * vol_ma_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(atr_4h_raw[i]) or
            np.isnan(volume_spike_12h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_12h_aligned[i]
        atr_val = atr_4h_raw[i]
        adx_cond = adx_12h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3
            if price < camarilla_s3[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3
            if price > camarilla_r3[i]:
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
            # LONG: Price breaks above Camarilla R3 AND ADX>25 AND volume spike
            if price > camarilla_r3[i] and adx_cond and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND ADX>25 AND volume spike
            elif price < camarilla_s3[i] and adx_cond and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_CamarillaR3S3_12hADX_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0