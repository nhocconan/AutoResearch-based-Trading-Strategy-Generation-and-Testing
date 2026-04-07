#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w ADX trend filter
# Uses Donchian channel breakouts for trend entry, confirmed by volume spike and ADX>25 for trend strength.
# Position size scaled by volatility (ATR ratio) to reduce exposure in high volatility.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at channel extremes.

name = "12h_donchian20_1d_volume_1w_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 1w ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    # Smooth TR, DM+
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr_14
    minus_di = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # ATR(14) for volatility scaling
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: scale position size inversely with volatility
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_scale = np.clip(1.0 / vol_ratio, 0.5, 1.5)  # scale between 0.5 and 1.5
        base_size = 0.25
        
        # Breakout conditions
        bullish_breakout = close[i] > highest_high[i-1]  # break above prior high
        bearish_breakout = close[i] < lowest_low[i-1]    # break below prior low
        
        # Volume confirmation: current volume > 1.5x average daily volume
        vol_confirm = volume[i] > 1.5 * avg_vol_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Long: bullish breakout + volume confirmation + strong trend
        if bullish_breakout and vol_confirm and strong_trend:
            signals[i] = base_size * vol_scale
        # Short: bearish breakout + volume confirmation + strong trend
        elif bearish_breakout and vol_confirm and strong_trend:
            signals[i] = -base_size * vol_scale
        else:
            signals[i] = 0.0
    
    return signals