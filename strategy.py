#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout + 1d volume spike + 1d ADX trend filter
# Long when: BB width at 20-period low (squeeze) + price breaks above upper BB + 1d volume > 2x 20-period MA + 1d ADX > 25
# Short when: BB width at 20-period low (squeeze) + price breaks below lower BB + 1d volume > 2x 20-period MA + 1d ADX > 25
# Exit when: price reverts to middle BB (20-period SMA) OR BB width expands above 50% percentile
# Uses Bollinger Bands for volatility contraction/expansion, volume for conviction, ADX for trend strength
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_BBSqueeze_Breakout_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 4h (20, 2)
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
        
        # BB width percentile lookback (50 periods) for squeeze detection
        bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
            lambda x: np.percentile(x, 20) if len(x) == 50 else np.nan, raw=False
        ).values
        bb_squeeze = bb_width < bb_width_percentile  # width in lowest 20%
        
        # Breakout conditions
        breakout_up = close > upper_bb
        breakout_down = close < lower_bb
        reversion_to_mean = (close > sma_20 - 0.1 * std_20) & (close < sma_20 + 0.1 * std_20)  # within 10% of middle
        bb_expansion = bb_width > pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
            lambda x: np.percentile(x, 50) if len(x) == 50 else np.nan, raw=False
        ).values  # width above 50th percentile
    else:
        sma_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        bb_squeeze = np.zeros(n, dtype=bool)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        reversion_to_mean = np.zeros(n, dtype=bool)
        bb_expansion = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data
        return np.zeros(n)
    
    # Calculate 1d volume MA (20-period)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    else:
        volume_spike_1d = np.zeros(len(vol_1d), dtype=bool)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        # ADX trend strength
        adx_strong = adx > 25
        adx_weak = adx < 20
    else:
        adx_strong = np.zeros(len(high_1d), dtype=bool)
        adx_weak = np.zeros(len(high_1d), dtype=bool)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(bb_width[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(adx_strong_aligned[i]) or np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: BB squeeze + breakout up + volume spike + strong ADX
            if (bb_squeeze[i] and breakout_up[i] and 
                volume_spike_1d_aligned[i] == 1.0 and 
                adx_strong_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: BB squeeze + breakout down + volume spike + strong ADX
            elif (bb_squeeze[i] and breakout_down[i] and 
                  volume_spike_1d_aligned[i] == 1.0 and 
                  adx_strong_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: mean reversion OR BB expansion
            if (reversion_to_mean[i] or bb_expansion[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: mean reversion OR BB expansion
            if (reversion_to_mean[i] or bb_expansion[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals