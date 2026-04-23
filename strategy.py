#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Donchian(20) breakout provides clear entry/exit signals with proven edge on SOLUSDT
- 1d EMA50 as trend filter (long only above, short only below) avoids whipsaw in ranging markets
- Volume > 2.0x 20-period average for confirmation to filter low-quality breakouts
- Position size: 0.30 discrete level to balance return and drawdown
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
"""

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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Donchian channels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) from prior 1d bar
    # Upper = max(high of last 20 days)
    # Lower = min(low of last 20 days)
    donch_h_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_l_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (using completed 1d bar)
    donch_h_20_aligned = align_htf_to_ltf(prices, df_1d, donch_h_20)
    donch_l_20_aligned = align_htf_to_ltf(prices, df_1d, donch_l_20)
    
    # 1d data for EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_h_20_aligned[i]) or
            np.isnan(donch_l_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donch_h_20_aligned[i]  # Close above upper band
        breakout_down = close[i] < donch_l_20_aligned[i]  # Close below lower band
        
        if position == 0:
            # Long: Donchian upper breakout AND price above 1d EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Donchian lower breakdown AND price below 1d EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakdown OR price crosses below 1d EMA50
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Donchian upper breakout OR price crosses above 1d EMA50
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0