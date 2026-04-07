#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend and Volume Confirmation.
Long when price breaks above Donchian(20) high with 1w uptrend and volume confirmation.
Short when price breaks below Donchian(20) low with 1w downtrend and volume confirmation.
Volume must be above 20-period average to confirm breakout.
Exit when price crosses Donchian middle or trend reverses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)
    
    # === 1D DONCHIAN CHANNEL (LTF) ===
    donch_length = 20
    donch_high = pd.Series(high).rolling(window=donch_length, min_periods=donch_length).max().values
    donch_low = pd.Series(low).rolling(window=donch_length, min_periods=donch_length).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(one_w_ema_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > one_w_ema_aligned[i]
        downtrend = close[i] < one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle OR trend turns down
            if close[i] < donch_mid[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR trend turns up
            if close[i] > donch_mid[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if close[i] > donch_high[i] and uptrend:
                # Breakout above upper band in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and downtrend:
                # Breakdown below lower band in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals