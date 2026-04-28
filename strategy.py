#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with volume spike and price > 1w EMA200.
# Short when price breaks below Donchian lower band with volume spike and price < 1w EMA200.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 to balance return and drawdown. Discrete levels minimize fee churn.
# Trend filter uses weekly EMA200 to avoid whipsaws in sideways markets and capture major trends.
# Targets ~20-50 trades/year on BTC/ETH/SOL.

name = "4h_Donchian20_1wEMA200_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient history for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA200 direction (price above/below EMA200)
        price_above_ema = close[i] > ema_200_1w_aligned[i]
        price_below_ema = close[i] < ema_200_1w_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > high_ma_20[i] and volume_spike[i]
        short_breakout = close[i] < low_ma_20[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < low_ma_20[i] or close[i] < ema_200_1w_aligned[i]
        short_exit = close[i] > high_ma_20[i] or close[i] > ema_200_1w_aligned[i]
        
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