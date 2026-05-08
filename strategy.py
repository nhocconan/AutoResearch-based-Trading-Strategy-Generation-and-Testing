#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily volume confirmation and weekly volatility filter
# Long when price breaks above weekly Donchian upper band, daily volume > 20-period average, and weekly ATR ratio < 0.5 (low volatility)
# Short when price breaks below weekly Donchian lower band, daily volume > 20-period average, and weekly ATR ratio < 0.5 (low volatility)
# Exit when price reverses back inside the weekly Donchian channel
# Uses weekly structure for trend/filter and daily for entry timing to minimize trades and avoid overtrading
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_WeeklyDonchianBreakout_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for structure and filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly ATR for volatility filter (ATR ratio: current ATR / 20-period average ATR)
    tr1 = np.abs(weekly_high - weekly_low)
    tr2 = np.abs(np.diff(weekly_close, prepend=weekly_close[0]))
    tr3 = np.abs(np.roll(weekly_close, 1) - weekly_close)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Daily volume confirmation (volume > 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when volatility is low (ATR ratio < 0.5)
        vol_filter = atr_ratio_aligned[i] < 0.5
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high, volume confirmation, low volatility
            if close[i] > donchian_high_aligned[i] and volume[i] > vol_ma[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low, volume confirmation, low volatility
            elif close[i] < donchian_low_aligned[i] and volume[i] > vol_ma[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks back below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks back above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals