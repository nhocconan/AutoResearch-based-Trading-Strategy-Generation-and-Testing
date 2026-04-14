# 11-07-2025
# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter
# Works in bull/bear by only taking breakouts in direction of ADX trend
# Low turnover: ~25-40 trades/year per target

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, window=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high, prepend=high[0]))
    tr2 = np.abs(np.diff(low, prepend=low[0]))
    tr3 = np.abs(np.diff(close, prepend=close[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=window, min_periods=window).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=window, min_periods=window).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
    return adx

def donchian_channels(high, low, window):
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    adx = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_strong = adx > 25  # Strong trend when ADX > 25
    adx_weak = adx < 20    # Weak trend when ADX < 20 (for exit)
    
    # Align ADX signals to 4h
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    # Donchian channels (20-period)
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned values
        strong_trend = adx_strong_aligned[i]
        weak_trend = adx_weak_aligned[i]
        
        if np.isnan(strong_trend) or np.isnan(weak_trend):
            continue
        
        if position == 0:
            # Enter long: break above upper + volume spike + strong uptrend
            if close[i] > upper[i] and volume[i] > vol_ma[i] * 1.5 and strong_trend > 0.5:
                position = 1
                signals[i] = position_size
            # Enter short: break below lower + volume spike + strong downtrend
            elif close[i] < lower[i] and volume[i] > vol_ma[i] * 1.5 and strong_trend > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: trend weakens or price hits opposite band
            if weak_trend > 0.5 or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: trend weakens or price hits opposite band
            if weak_trend > 0.5 or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0