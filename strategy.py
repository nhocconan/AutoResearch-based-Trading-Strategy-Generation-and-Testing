#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Long when price breaks above upper Donchian channel AND 1w ADX > 25 (strong trend) AND volume > 1.5 * 24-bar avg volume
# Short when price breaks below lower Donchian channel AND 1w ADX > 25 AND volume > 1.5 * 24-bar avg volume
# Exit when price retraces to the Donchian midpoint (mean reversion)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1w ADX provides strong trend filter to avoid whipsaws in ranging markets
# Volume threshold set to 1.5x to reduce false breakouts while maintaining sufficient trade frequency

name = "12h_Donchian20_1wADX25_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 12h timeframe (based on previous 20 bars)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    prev_high_20 = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20 = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_upper = prev_high_20
    donchian_lower = prev_low_20
    donchian_mid = (prev_high_20 + prev_low_20) / 2.0
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1w).rolling(2).max().values - pd.Series(low_1w).rolling(2).min().values
    tr2 = np.abs(pd.Series(high_1w).shift(1).values - pd.Series(close_1w).values)
    tr3 = np.abs(pd.Series(low_1w).shift(1).values - pd.Series(close_1w).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff().values
    down_move = -pd.Series(low_1w).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (np.abs(plus_di) + np.abs(minus_di) + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate volume confirmation: volume > 1.5 * 24-bar average volume
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper channel AND strong trend AND volume spike
            if close[i] > donchian_upper[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower channel AND strong trend AND volume spike
            elif close[i] < donchian_lower[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to midpoint (mean reversion)
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals