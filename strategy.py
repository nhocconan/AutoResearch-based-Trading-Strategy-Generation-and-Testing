#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + 1w choppiness regime filter
# - Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND weekly chop < 61.8 (trending regime)
# - Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND weekly chop < 61.8 (trending regime)
# - Exit when price crosses Donchian(20) midline
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation reduces false breakouts
# - Weekly choppiness filter ensures we only trade in trending markets (avoids choppy/range-bound periods)

name = "12h_1d_1w_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d close for weekly alignment
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w choppiness regime (CHOP(14))
    def true_range(high, low, close_prev):
        return np.maximum(np.maximum(high - low, np.abs(high - close_prev)), np.abs(low - close_prev))
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate TR for 1w
    close_prev_1w = np.roll(close_1w, 1)
    close_prev_1w[0] = close_1w[0]  # first period
    tr_1w = true_range(high_1w, low_1w, close_prev_1w)
    
    # Calculate +DM and -DM for 1w
    high_diff = np.diff(high_1w, prepend=high_1w[0])
    low_diff = np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # smoothed TR, +DM, -DM (14-period)
    def smoothed_avg(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + (arr[i] / period)
        return result
    
    tr14 = smoothed_avg(tr_1w, 14)
    plus_dm14 = smoothed_avg(plus_dm, 14)
    minus_dm14 = smoothed_avg(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # Calculate DX and CHOP
    dx14 = np.where((plus_di14 + minus_di14) != 0, 
                    np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 
                    0)
    
    # ADX (smoothed DX)
    def smoothed_avg_adx(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    adx14 = smoothed_avg_adx(dx14, 14)
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14)/(ATR14 * n)) / log10(n)
    # Simplified: CHOP = 100 * log10(sum(TR14)/ (ATR14 * 14)) / log10(14)
    # We'll use a proxy: CHOP ≈ 100 * (1 - ADX/100) when ADX < 100, else 0
    # Better approximation: CHOP = 100 * log10(14) / log10(sum(TR14)/atr14) but we simplify
    # Using: CHOP = 50 + 50 * (1 - ADX/25) for ADX < 25, else 50 * (75/ADX) for ADX >= 25
    chop = np.where(adx14 < 25,
                    50 + 50 * (1 - adx14/25),
                    50 * (75 / np.maximum(adx14, 1)))
    
    # Align HTF indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND volume spike AND chop < 61.8 (trending)
            if (close[i] > donch_high[i] and 
                volume_spike_aligned[i] and 
                chop_aligned[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND volume spike AND chop < 61.8 (trending)
            elif (close[i] < donch_low[i] and 
                  volume_spike_aligned[i] and 
                  chop_aligned[i] < 61.8):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midline
            exit_long = (position == 1 and close[i] < donch_mid[i])
            exit_short = (position == -1 and close[i] > donch_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals