#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ADX regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-bar avg AND 1d ADX(14) > 25 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-bar avg AND 1d ADX(14) > 25 (trending)
# - Exit when price crosses Donchian(20) midpoint (mean reversion structure)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures structural breaks; volume confirms institutional interest; ADX filters ranging markets
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets via breakouts, bear markets via breakdowns, avoids ranging periods

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg)
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_trending = adx > 25
    
    # Align 1d indicators to 4h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    # Pre-compute Donchian(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Breakout conditions
    breakout_up = close > donchian_high
    breakout_down = close < donchian_low
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_aligned[i]) or np.isnan(adx_trending_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1d volume spike AND 1d trending
            if (breakout_up[i] and 
                vol_spike_aligned[i] and 
                adx_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1d volume spike AND 1d trending
            elif (breakout_down[i] and 
                  vol_spike_aligned[i] and 
                  adx_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian midpoint
            # Exit when price crosses Donchian midpoint (mean reversion structure)
            if position == 1 and close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals