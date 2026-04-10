#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 1d EMA13)
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# - Short when Bull Power < 0 AND Bear Power > 0 AND 1d ADX > 25 (strong trend)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - ADX regime filter ensures we only trade in strong trending markets, reducing whipsaws
# - Elder Ray captures the power of bulls/bears relative to the 13-period EMA

name = "6h_1d_elder_ray_adx_regime_v1"
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
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX(14) for regime filter
    # True Range
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = np.nan  # First value has no previous close
    low_close[0] = np.nan
    tr = np.nanmax(np.column_stack([high_low, high_close, low_close]), axis=1)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and ATR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(atr_14_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_14_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray components
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # Align HTF indicators to LTF
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when trend weakens or power shifts
            if adx_14_1d_aligned[i] < 20 or bull_power_1d_aligned[i] <= 0 or bear_power_1d_aligned[i] >= 0:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when trend weakens or power shifts
            if adx_14_1d_aligned[i] < 20 or bull_power_1d_aligned[i] >= 0 or bear_power_1d_aligned[i] <= 0:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when bulls are in control and trend is strong
            if (bull_power_1d_aligned[i] > 0 and 
                bear_power_1d_aligned[i] < 0 and 
                adx_14_1d_aligned[i] > 25):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Enter short when bears are in control and trend is strong
            elif (bull_power_1d_aligned[i] < 0 and 
                  bear_power_1d_aligned[i] > 0 and 
                  adx_14_1d_aligned[i] > 25):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals