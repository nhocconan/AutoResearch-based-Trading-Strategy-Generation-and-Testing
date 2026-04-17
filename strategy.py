#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d ADX trend filter.
- Long when price breaks above 4h Donchian upper (20) + volume > 1.8x 20-period 1h volume MA + 1d ADX > 25
- Short when price breaks below 4h Donchian lower (20) + volume > 1.8x 20-period 1h volume MA + 1d ADX > 25
- Fixed position size 0.20 to manage drawdown
- Uses proven edge: Donchian breakouts (trend following) + volume spike + HTF ADX trend strength
- Designed for 1h timeframe with strict entry conditions to limit trades to 60-150 total over 4 years
- Session filter (08-20 UTC) to avoid low-liquidity periods
- Works in bull markets (buying breakouts with strong uptrend) and bear markets (selling breakdowns with strong downtrend)
- ADX filter ensures we only trade when there is sufficient trend strength, reducing whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (HTF for structure)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    donchian_high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ADX trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / atr_14
    di_minus = 100 * dm_minus_14 / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) on 1h for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to primary timeframe (1h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Session filter: 08-20 UTC (intraday active hours)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d ADX > 25 (trend strength)
            # Long: price breaks above Donchian high + volume spike + ADX > 25
            if price > donchian_high and vol > 1.8 * vol_ma and adx_val > 25:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low + volume spike + ADX > 25
            elif price < donchian_low and vol > 1.8 * vol_ma and adx_val > 25:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below Donchian low (trend reversal) or ADX < 20 (trend weakening)
            if price < donchian_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit on close above Donchian high (trend reversal) or ADX < 20 (trend weakening)
            if price > donchian_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_VolumeSpike_1dADX25_SessionFilter"
timeframe = "1h"
leverage = 1.0