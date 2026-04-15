#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation
# Long when price breaks above Camarilla R3 (1d) + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S3 (1d) + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla levels from higher timeframe (1d) provide structure-based breakout points that work in ranging markets.
# ADX filter ensures we only trade in trending regimes, reducing whipsaws in choppy markets.
# Volume threshold (1.5x) targets ~20-30 trades/year on 4h timeframe to avoid overtrading.
# This combines proven patterns: Camarilla pivot breakouts (ETH Sharpe 1.47) + ADX regime filter + volume confirmation.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) and ADX ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    # Calculate Camarilla levels (using previous day's data)
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    rng = high_1d[-1] - low_1d[-1]
    camarilla_r3 = pp + (rng * 1.1 / 4.0)
    camarilla_s3 = pp - (rng * 1.1 / 4.0)
    
    # Since Camarilla levels are based on previous day, we need to shift them
    # and align to 4h timeframe
    camarilla_r3_series = pd.Series([camarilla_r3] * len(df_1d)).shift(1)  # Use previous day's levels
    camarilla_s3_series = pd.Series([camarilla_s3] * len(df_1d)).shift(1)
    
    # Handle NaN from shift
    camarilla_r3_series.iloc[0] = camarilla_r3  # First day uses same level
    camarilla_s3_series.iloc[0] = camarilla_s3
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_series.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_series.values)
    
    # === 1d ADX (14-period) for regime filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - close_1d.shift(1)))
    tr3 = pd.Series(np.abs(low_1d - close_1d.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = high_1d.diff()
    down_move = low_1d.shift(1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth the DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr
    
    # Calculate DX and ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # === 4h Donchian Channel (20-period) for entry timing ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # ADX(14) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Regime filter: ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Trending regime (ADX > 25)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and \
           trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Trending regime (ADX > 25)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and \
             trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_1dADX_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0