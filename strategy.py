#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above Camarilla R3 level + 1d volume > 1.5x 20-period avg + 1d ADX < 25 (range regime)
# Short when price breaks below Camarilla S3 level + 1d volume > 1.5x 20-period avg + 1d ADX < 25 (range regime)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla pivots provide intraday support/resistance levels that work well in ranging markets.
# Volume filter ensures breakouts have conviction, ADX < 25 filters for ranging conditions where mean reversion works.
# Designed to generate ~20-40 trades/year on 12h timeframe to avoid overtrading and fee drag.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Volume SMA and ADX ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Volume SMA (20-period)
    vol_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # ADX (14-period)
    # Calculate True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di_14_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14_1d
    minus_di_14_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14_1d
    
    # DX and ADX
    dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d)
    adx_14_1d = pd.Series(dx_14_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # === 12h Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from previous 1d OHLC
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # We use the 1d OHLC to calculate levels for the 12h chart
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # ADX + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period 1d volume SMA
        vol_confirm = volume[i] > (vol_sma_20_1d_aligned[i] * 1.5)
        
        # Regime filter: 1d ADX < 25 (range-bound market)
        range_regime = adx_14_1d_aligned[i] < 25
        
        # Only trade in range regime
        if not range_regime:
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 level
        # 2. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 level
        # 2. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolume_ADX_RangeFilter_v1"
timeframe = "12h"
leverage = 1.0