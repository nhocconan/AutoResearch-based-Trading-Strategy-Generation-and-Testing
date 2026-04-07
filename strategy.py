#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action strategy using 1-day VWAP as dynamic support/resistance
# Long when price crosses above VWAP with rising volume and bullish 1-week trend (ADX > 20)
# Short when price crosses below VWAP with rising volume and bearish 1-week trend (ADX > 20)
# Exit when price returns to VWAP or stoploss hits (2.0 * ATR)
# Uses 1-day VWAP for mean reversion target and 1-week ADX for trend filter
# Designed to work in both bull (trend following) and bear (mean reversion) markets
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_vwap_1d_vol_1w_adx_v1"
timeframe = "12h"
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
    
    # 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-day VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    volume_1d = df_1d['volume'].values
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / (vwap_den + 1e-10)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 1-week ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14 = adx[-1]  # Current ADX value
    adx_array = np.full(len(close_1w), adx_14)  # Constant array for alignment
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_array)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to VWAP (mean reversion)
            elif close[i] <= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to VWAP (mean reversion)
            elif close[i] >= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VWAP cross with volume confirmation and ADX filter
            # Volume filter: current volume > 1.5 * 20-period average volume
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
            volume_filter = volume[i] > 1.5 * vol_ma
            # Trend filter: weekly ADX > 20
            trend_filter = adx_aligned[i] > 20
            
            # Long: price crosses above VWAP + volume filter + trend filter
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below VWAP + volume filter + trend filter
            elif close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals