#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and ADX regime filter
# Long when price breaks above Camarilla R3 level + ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S3 level + ADX > 25 + volume > 1.5x 20-period avg
# Uses 1d Camarilla levels calculated from prior 1d OHLC, aligned to 6h bars
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla pivots work well in ranging markets; ADX filter ensures we only trade in trending conditions
# Volume confirmation reduces false breakouts

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
    
    # Get 1d HTF data once before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    # Camarilla levels based on prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R4, R3, S3, S4
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Indicator: ADX (14-period) for trend strength ===
    # ADX calculation: +DI, -DI, then DX, then smoothed ADX
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
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d data for Camarilla and ADX (14+14+14 = ~42 periods) + volume(20) + buffer
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 level
        # 2. ADX > 25 (trending market)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and \
           trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 level
        # 2. ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and \
             trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_1dADX_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0