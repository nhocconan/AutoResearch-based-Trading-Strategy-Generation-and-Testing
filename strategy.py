#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation.
# In bull markets: buy breakouts above upper band in uptrend.
# In bear markets: sell breakouts below lower band in downtrend.
# Volume confirms breakout strength. Weekly trend prevents counter-trend entries.
# Target: 10-25 trades/year, low frequency to minimize fee drag.

name = "1d_Donchian20_1wTrend_Volume_V1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend (needs only completed weekly bar)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            signals[i] = 0.0
            continue
            
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
        
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long: breakout up + weekly uptrend + volume
            if breakout_up and ema_34_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False and vol_confirm:
                # Simplified: use aligned weekly EMA vs current weekly close approximation
                # Since we can't get weekly close easily, use price vs EMA
                if close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: breakout down + weekly downtrend + volume
            elif breakout_down and close[i] < ema_34_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: breakdown below lower Donchian or trend reversal
            if close[i] < low_20[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: breakout above upper Donchian or trend reversal
            if close[i] > high_20[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals