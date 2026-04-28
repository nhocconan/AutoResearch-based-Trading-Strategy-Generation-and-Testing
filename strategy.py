#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses 12h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Donchian(20) from 1d: long when price breaks above upper band with volume, short when breaks below lower band.
# 1d EMA34 provides trend filter: long only when price > EMA34, short only when price < EMA34.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull and bear markets via trend filter + breakout logic.

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian(20) and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    donchian_hi = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lo = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    donchian_hi_aligned = align_htf_to_ltf(prices, df_1d, donchian_hi)
    donchian_lo_aligned = align_htf_to_ltf(prices, df_1d, donchian_lo)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_hi_aligned[i]) or np.isnan(donchian_lo_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_hi_aligned[i] and volume_spike[i]
        short_breakout = close[i] < donchian_lo_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < donchian_lo_aligned[i] or close[i] < ema_34_1d_aligned[i]
        short_exit = close[i] > donchian_hi_aligned[i] or close[i] > ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals