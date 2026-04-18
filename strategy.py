# 4h_Camarilla_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict
# Hypothesis: Camarilla pivot levels (R1/S1) from 1-day act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and range filter (low ADX) indicate momentum.
# Works in bull/bear: captures breakouts in trending markets while avoiding false signals in ranging markets via ADX filter.
# Target: 20-40 trades/year to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) for each daily bar
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Calculate ADX (14) on 1-day for range filter
    # +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.append([np.nan], close_1d[:-1]))
    tr3 = np.abs(low_1d - np.append([np.nan], close_1d[:-1]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align R1, S1, and ADX to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike on 4h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(adx_4h[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: only trade when ADX < 25 (ranging/mild trend) to avoid whipsaws
        range_filter = adx_4h[i] < 25
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and range filter
            if close[i] > r1_4h[i] and volume_spike[i] and range_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and range filter
            elif close[i] < s1_4h[i] and volume_spike[i] and range_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 OR ADX rises above 25 (trending) OR volume drops
            if (close[i] < s1_4h[i]) or (adx_4h[i] >= 25) or (not volume_spike[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 OR ADX rises above 25 OR volume drops
            if (close[i] > r1_4h[i]) or (adx_4h[i] >= 25) or (not volume_spike[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "4h"
leverage = 1.0