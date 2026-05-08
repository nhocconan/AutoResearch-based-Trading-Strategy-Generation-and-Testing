# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX-based trend strength with 12h Donchian breakout and volume confirmation.
# Long when ADX > 25 (trending), price breaks above 12h Donchian upper band (20), and 6h volume > 1.5x 20-period average.
# Short when ADX > 25, price breaks below 12h Donchian lower band (20), and 6h volume > 1.5x 20-period average.
# Exit when ADX < 20 (trend weak) or opposite Donchian breakout occurs.
# Uses ADX for trend filter to avoid whipsaws in ranging markets, Donchian for breakout signals, volume for confirmation.
# Target: 80-160 total trades over 4 years (20-40/year) for balanced risk/reward.

name = "6h_ADX_DonchianBreakout_12hVolume"
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
    
    # 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 6h data
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(low)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Sufficient warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX > 25, price breaks above Donchian high, volume spike
            long_cond = (adx[i] > 25) and (close[i] > donchian_high_aligned[i]) and volume_filter[i]
            # Short conditions: ADX > 25, price breaks below Donchian low, volume spike
            short_cond = (adx[i] > 25) and (close[i] < donchian_low_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (trend weak) or price breaks below Donchian low
            if adx[i] < 20 or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (trend weak) or price breaks above Donchian high
            if adx[i] < 20 or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals