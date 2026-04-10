#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ADX regime filter
# - Primary: 4h price breaks above Camarilla H3 level (long) or below L3 level (short)
# - Volume filter: 1d volume > 1.5x 20-period volume MA to confirm institutional participation
# - Regime filter: 1d ADX(14) > 25 to ensure trending market (avoid ranging conditions)
# - Exit: Price crosses Camarilla pivot point (mean reversion exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# - Works in bull/bear: Camarilla levels adapt to volatility, ADX filter avoids whipsaws, volume confirms breakouts

name = "4h_1d_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous 1d bar) on 4h
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3_level = prev_close_1d + 1.1 * camarilla_range / 2
    l3_level = prev_close_1d - 1.1 * camarilla_range / 2
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    
    # Align HTF levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_level)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    # Calculate 1d ADX(14) for regime filter
    # +DM = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
    # -DM = max(prev_low - low, 0) if > max(high - prev_high, 0) else 0
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = np.where(atr_14 > 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 > 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume MA(20) for volume filter
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period volume MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmed = volume_1d_aligned[i] > 1.5 * volume_ma_20_aligned[i]
        
        # Regime filter: ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: breakout above H3 + volume confirmation + trending regime
            if (close[i] > h3_aligned[i] and volume_confirmed and trending):
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below L3 + volume confirmation + trending regime
            elif (close[i] < l3_aligned[i] and volume_confirmed and trending):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Camarilla pivot point (mean reversion exit)
            if position == 1:  # Long position
                if close[i] < pivot_aligned[i]:  # Exit when price breaks below pivot
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > pivot_aligned[i]:  # Exit when price breaks above pivot
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals