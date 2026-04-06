#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s ADX + volume regime filter for trend strength.
# Uses daily ADX to determine if market is trending (ADX > 25) or ranging (ADX < 20).
# In trending regime: enter long on breakout above 6h high with volume > 1.5x average.
# In ranging regime: enter long at 6h low with volume > average, short at 6h high.
# Uses weekly EMA trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_adx_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1w EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h price channels
    high_6h = pd.Series(high).rolling(window=2, min_periods=2).max().values  # Previous bar high
    low_6h = pd.Series(low).rolling(window=2, min_periods=2).min().values    # Previous bar low
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma  # Volume above average
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume
    
    # ATR for stoploss
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(high_6h[i]) or np.isnan(low_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime determination
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions
            elif (is_trending and close[i] < ema_1w_aligned[i]) or \
                 (is_ranging and close[i] > high_6h[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions
            elif (is_trending and close[i] > ema_1w_aligned[i]) or \
                 (is_ranging and close[i] < low_6h[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if is_trending and vol_strong[i]:
                # Trending regime: breakout entries
                if close[i] > high_6h[i]:  # Break above recent high
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_6h[i]:  # Break below recent low
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif is_ranging and vol_filter[i]:
                # Ranging regime: mean reversion at extremes
                if close[i] <= low_6h[i] * 1.001 and close[i] >= low_6h[i] * 0.999:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] >= high_6h[i] * 0.999 and close[i] <= high_6h[i] * 1.001:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals