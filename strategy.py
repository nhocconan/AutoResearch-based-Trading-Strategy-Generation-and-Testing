#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout for direction + 1d ADX trend filter + volume spike confirmation
# 4h Donchian(20) provides structural breakout signals (works in bull/bear via trend filter)
# 1d ADX > 25 ensures we only trade in trending markets, avoiding choppy conditions
# Volume confirmation: 1h volume > 2.0x 20-period average to filter weak breakouts
# Fixed position size 0.20 to control risk and minimize fee churn
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4h Donchian Channel (20-period) for breakout direction ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high/low: highest high/lowest low of last 20 periods
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (waits for completed 4h bar)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # === 1d ADX (14-period) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        adx = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        vol_confirm = volume[i] > vol_ma * 2.0  # Volume spike: 2x average
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low OR ADX weakens (< 20)
            if price < donch_low or adx < 20:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high OR ADX weakens (< 20)
            if price > donch_high or adx < 20:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND ADX > 25 AND volume confirmation
            if price > donch_high and adx > 25 and vol_confirm:
                signals[i] = 0.20
                position = 1
                continue
            # Short when: price breaks below Donchian low AND ADX > 25 AND volume confirmation
            elif price < donch_low and adx > 25 and vol_confirm:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_1dADX25_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0