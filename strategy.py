#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d ADX25 regime filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX<25),
# fade extremes for mean reversion. In trending markets (ADX>=25), breakouts continue.
# Volume spike (>1.8x 20-bar average) confirms momentum. Uses discrete position sizing
# (0.25) to minimize fee churn. Works in both bull and bear via ADX regime adaptation.

name = "6h_WilliamsR_Extreme_1dADX25_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX25 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX25 for regime filter
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_25_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r_values = williams_r.values
    
    # Calculate 6h volume spike: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_25_aligned[i]) or 
            np.isnan(williams_r_values[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX >= 25 = trending, ADX < 25 = ranging
        is_trending = adx_25_aligned[i] >= 25
        is_ranging = adx_25_aligned[i] < 25
        
        # Williams %R extremes
        oversold = williams_r_values[i] <= -80
        overbought = williams_r_values[i] >= -20
        
        # Mean reversion in ranging markets: fade extremes
        long_mr = is_ranging and oversold and volume_spike[i]
        short_mr = is_ranging and overbought and volume_spike[i]
        
        # Breakout continuation in trending markets: extremes as momentum
        long_break = is_trending and oversold and volume_spike[i]  # Oversold in uptrend = continuation
        short_break = is_trending and overbought and volume_spike[i]  # Overbought in downtrend = continuation
        
        # Exit conditions: opposite extreme or regime change
        long_exit = williams_r_values[i] >= -20 or (is_trending and not is_trending)  # Overbought or regime to ranging
        short_exit = williams_r_values[i] <= -80 or (is_trending and not is_trending)  # Oversold or regime to ranging
        
        # Handle entries and exits
        if (long_mr or long_break) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_mr or short_break) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals