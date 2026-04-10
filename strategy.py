#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Primary signal: 6h Elder Ray (Bull Power/Bear Power) cross zero for entry
# - Regime filter: 1d ADX > 25 to ensure trending market (avoid whipsaws in ranges)
# - Volume confirmation: 6h volume > 1.5x 20-period average volume (avoid low-participation signals)
# - Works in bull/bear: In strong trends (ADX > 25), Elder Ray captures acceleration; in weak trends (ADX < 20), strategy stays flat
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(20) on 6h

name = "6h_1d_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI, -DI, DX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h Elder Ray
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_6h - ema_13
    bear_power = low_6h - ema_13
    
    # Pre-compute 6h ATR(20) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_13[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power crosses above zero OR stoploss hit
            if bear_power[i] > 0 or close_6h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power crosses below zero OR stoploss hit
            if bull_power[i] < 0 or close_6h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with ADX regime filter and volume confirmation
            # Only trade in trending markets (ADX > 25)
            if adx_aligned[i] > 25 and volume_spike[i]:
                # Long: Bull Power crosses above zero (bulls taking control)
                if bull_power[i] > 0 and bull_power[i-1] <= 0:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bear Power crosses below zero (bears taking control)
                elif bear_power[i] < 0 and bear_power[i-1] >= 0:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals