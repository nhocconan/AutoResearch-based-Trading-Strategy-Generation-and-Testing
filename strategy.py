#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout + Volume Spike + Weekly Trend Filter
# Uses daily Donchian channel breakouts with volume confirmation and weekly trend strength filter
# Weekly ADX > 25 ensures we only trade in trending markets on higher timeframe, avoiding false breakouts
# Works in bull/bear by capturing breakouts in the direction of the weekly trend
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly ADX (14-period) on weekly data
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # True Range for weekly
    wtr1 = whigh - wlow
    wtr2 = np.abs(whigh - np.roll(wclose, 1))
    wtr3 = np.abs(wlow - np.roll(wclose, 1))
    wtr2[0] = 0
    wtr3[0] = 0
    wtr = np.maximum(wtr1, np.maximum(wtr2, wtr3))
    
    # Directional Movement for weekly
    wdm_plus = np.where((whigh - np.roll(whigh, 1)) > (np.roll(wlow, 1) - wlow), 
                        np.maximum(whigh - np.roll(whigh, 1), 0), 0)
    wdm_minus = np.where((np.roll(wlow, 1) - wlow) > (whigh - np.roll(whigh, 1)), 
                         np.maximum(np.roll(wlow, 1) - wlow, 0), 0)
    wdm_plus[0] = 0
    wdm_minus[0] = 0
    
    # Smoothed values for weekly
    wtr14 = pd.Series(wtr).rolling(window=14, min_periods=14).sum().values
    wdm_plus14 = pd.Series(wdm_plus).rolling(window=14, min_periods=14).sum().values
    wdm_minus14 = pd.Series(wdm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators for weekly
    wdi_plus = np.where(wtr14 != 0, 100 * wdm_plus14 / wtr14, 0)
    wdi_minus = np.where(wtr14 != 0, 100 * wdm_minus14 / wtr14, 0)
    
    # DX and ADX for weekly
    wdx = np.where((wdi_plus + wdi_minus) != 0, 100 * np.abs(wdi_plus - wdi_minus) / (wdi_plus + wdi_minus), 0)
    wadx = pd.Series(wdx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe (with proper delay for weekly bar close)
    wadx_aligned = align_htf_to_ltf(prices, df_1w, wadx)
    
    # Daily Donchian channel (20-period)
    d_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    d_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(d_high[i]) or np.isnan(d_low[i]) or
            np.isnan(wadx_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Weekly trend filter: only trade when weekly ADX > 25 (trending market)
        if wadx_aligned[i] < 25:
            # In weak trend/ranging market on weekly, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter
            if price > d_high[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter
            elif price < d_low[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if price < d_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if price > d_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Volume_WeeklyADX_Filter"
timeframe = "1d"
leverage = 1.0