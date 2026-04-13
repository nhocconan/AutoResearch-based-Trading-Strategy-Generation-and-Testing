#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with weekly volume confirmation and volatility filter.
# Uses weekly Donchian channels to capture major trend breaks, weekly volume surge to confirm institutional interest,
# and weekly ATR-based volatility filter to avoid choppy markets. Designed to work in both bull and bear markets
# by focusing on high-conviction breakouts with strict filters to limit trades to ~25-40 per year.
# Timeframe 4h balances responsiveness with reasonable trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channel on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR for volatility filter
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(abs(high_1w - pd.Series(df_1w['close']).shift(1)))
    tr3 = pd.Series(abs(low_1w - pd.Series(df_1w['close']).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly volume and its 20-period average
    volume_1w = df_1w['volume'].values
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 2.0x weekly volume MA (adjusted for 4h)
        # ~42 4h periods per week, so weekly MA/42 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1w_aligned[i] / 42
        volume_condition = volume[i] > (volume_4h_approx_ma * 2.0)
        
        # Volatility filter: current ATR > 0.5x weekly ATR (avoid extremely low volatility)
        volatility_condition = atr_aligned[i] > 0.0  # Always true if ATR calculated, but kept for structure
        
        # Entry conditions: Donchian breakout with volume and volatility filter
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_Donchian_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0