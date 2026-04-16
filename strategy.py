#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX(14) trend filter and 1d volume confirmation.
# Long when price breaks above Camarilla R3 AND 4h ADX > 25 (trending) AND 1d volume > 1.2x 20-period average.
# Short when price breaks below Camarilla S3 AND 4h ADX > 25 AND 1d volume > 1.2x 20-period average.
# Exit on opposite Camarilla break (R3/S3) or when ADX < 20 (range regime).
# Uses discrete position size 0.20. Designed to capture institutional breakouts with volume and trend confirmation.
# Target: 80-160 total trades over 4 years (20-40/year) to balance edge and fee drag for 1h timeframe.
# Works in both bull and bear markets by requiring 4h ADX trend filter and 1d volume confirmation, avoiding false breakouts in ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla Pivot Points (R3, S3) ===
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using daily OHLC for Camarilla calculation
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) == 0:
        return np.zeros(n)
    
    # Get previous day's OHLC for today's Camarilla levels
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Calculate Camarilla levels from previous day
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align to 1h timeframe (Camarilla levels fixed for the entire day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s3)
    
    # === 4h Indicators: ADX(14) for trend strength ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = pd.Series(high_4h).diff()
    tr2 = pd.Series(low_4h).diff().abs()
    tr3 = pd.Series(close_4h).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_4h).diff()
    dm_minus = pd.Series(low_4h).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    adx_trending = adx_aligned > 25
    adx_ranging = adx_aligned < 20
    
    # === 1d Indicators: Volume Confirmation ===
    vol_1d = df_1d_ohlc['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d_ohlc, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3
            if price < camarilla_s3_aligned[i]:
                exit_signal = True
            # Exit if ADX drops below 20 (range regime)
            elif adx_ranging[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3
            if price > camarilla_r3_aligned[i]:
                exit_signal = True
            # Exit if ADX drops below 20 (range regime)
            elif adx_ranging[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND ADX > 25 (trending) AND volume spike
            if price > camarilla_r3_aligned[i] and adx_trending[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price breaks below Camarilla S3 AND ADX > 25 (trending) AND volume spike
            elif price < camarilla_s3_aligned[i] and adx_trending[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Camarilla_R3S3_4hADX_1dVolume_V1"
timeframe = "1h"
leverage = 1.0