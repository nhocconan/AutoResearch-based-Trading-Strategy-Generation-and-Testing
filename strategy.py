#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# - Primary signal: Strong Bull Power (>0) + Bear Power weakening (<0) = long; vice versa for short
# - 1d regime filter: ADX(14) > 25 for trending markets (avoid chop)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(20) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Elder Ray captures momentum shifts; ADX filter avoids false signals in range

name = "6h_1d_elder_ray_adx_v1"
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
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    up_smoothed = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    down_smoothed = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    tr_smoothed = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * up_smoothed / tr_smoothed
    minus_di = 100 * down_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # ADX > 25 = trending regime
    adx_filter = adx > 25
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h ATR(20) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(adx_filter_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Elder Ray weakening OR stoploss hit
            bull_power = prices['high'].values[i] - ema_13[i]
            bear_power = ema_13[i] - prices['low'].values[i]
            if bull_power <= 0 or bear_power < 0 or close_6h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray weakening OR stoploss hit
            bull_power = prices['high'].values[i] - ema_13[i]
            bear_power = ema_13[i] - prices['low'].values[i]
            if bear_power <= 0 or bull_power < 0 or close_6h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with ADX filter
            if adx_filter_aligned[i]:
                bull_power = prices['high'].values[i] - ema_13[i]
                bear_power = ema_13[i] - prices['low'].values[i]
                
                # Long: Strong bull power, weakening bear power
                if bull_power > 0 and bear_power < 0:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Strong bear power, weakening bull power
                elif bear_power > 0 and bull_power < 0:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals