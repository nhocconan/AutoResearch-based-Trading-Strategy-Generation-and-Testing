#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe strategy using 4h Donchian breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h for signal direction (Donchian breakout) and 1d for trend filter (EMA34)
- 1h only for entry timing precision to reduce whipsaw
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Position size: 0.20 (discrete level to minimize fee churn)
- Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.8x 20-period average (slightly looser for 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h Donchian channels for trend direction (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    donch_hi_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lo_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (use prior completed 4h bar)
    donch_hi_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_hi_4h)
    donch_lo_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lo_4h)
    
    # 1d EMA34 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # EMA34, volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_hi_4h_aligned[i]) or
            np.isnan(donch_lo_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > donch_hi_4h_aligned[i-1]  # Close above prior 4h Donchian high
        breakout_down = close[i] < donch_lo_4h_aligned[i-1]  # Close below prior 4h Donchian low
        
        if position == 0:
            # Long: 4h Donchian breakout up AND price > 1d EMA34 AND volume confirmation AND in session
            if breakout_up and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h Donchian breakout down AND price < 1d EMA34 AND volume confirmation AND in session
            elif breakout_down and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h Donchian breakout down OR price < 1d EMA34 (trend flip)
            if breakout_down or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h Donchian breakout up OR price > 1d EMA34 (trend flip)
            if breakout_up or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Breakout_1dEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0