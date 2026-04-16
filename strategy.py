#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA trend filter and 1d ADX regime filter with volume spike confirmation.
# Long when: price > 4h EMA50, 1d ADX > 25 (trending), and volume > 2.0x 20-period MA.
# Short when: price < 4h EMA50, 1d ADX > 25 (trending), and volume > 2.0x 20-period MA.
# Exit when price crosses 4h EMA50 in opposite direction.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Position size 0.20 discrete levels to minimize fee churn.
# Target: 60-120 total trades over 4 years (15-30/year) to balance edge and fee drag.
# Works in bull/bear: ADX filter ensures we only trade strong trends, volume confirms conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute before loop)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Get 1d data once before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX (14) for regime filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di_14 = 100 * plus_dm_14 / atr_1d
    minus_di_14 = 100 * minus_dm_14 / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        ema_trend = ema_4h_50_aligned[i]
        adx = adx_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if price crosses below 4h EMA50
            if price < ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if price crosses above 4h EMA50
            if price > ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Regime filter: only trade when ADX > 25 (strong trend)
            regime_filter = adx > 25
            
            # Volume confirmation: volume > 2.0x 20-period MA
            vol_filter = vol > 2.0 * vol_ma
            
            # LONG: Price above 4h EMA50 in strong trend with volume confirmation
            if (price > ema_trend) and regime_filter and vol_filter:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price below 4h EMA50 in strong trend with volume confirmation
            elif (price < ema_trend) and regime_filter and vol_filter:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        signals[i] = position * 0.20
    
    return signals

name = "1h_4hEMA50_1dADX25_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0