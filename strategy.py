#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume confirmation and 1d ADX trend filter
# Long when price breaks above 1d Camarilla R3 + 4h volume > 1.5x 20-period avg + 1d ADX > 25
# Short when price breaks below 1d Camarilla S3 + 4h volume > 1.5x 20-period avg + 1d ADX > 25
# Uses 1h for entry timing, 4h/1d for signal direction/filters to reduce trades and avoid fee drag
# Discrete position sizing (0.20) to control drawdown. Target: 60-150 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 4h HTF data once before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Indicator: ADX (14-period) for trend strength ===
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # +DM and -DM
    up_move = high_1d_series.diff()
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    atr_smooth = atr.rolling(window=14, min_periods=14).mean()
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_smooth)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # === 4h Indicator: Volume SMA (20-period) for confirmation ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    vol_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period 4h volume SMA
        # We need to map 1h index to 4h volume data - use the aligned array directly
        vol_confirm = volume_4h[i // 4] > (vol_sma_20_4h[i // 4] * 1.5) if i // 4 < len(volume_4h) else False
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        if (close[i] > camarilla_r3_aligned[i]) and trending and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        elif (close[i] < camarilla_s3_aligned[i]) and trending and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_4hVol_1dADX_Filter_v1"
timeframe = "1h"
leverage = 1.0