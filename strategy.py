# 12h Donchian Breakout with Weekly Trend and Volume Confirmation
# Strategy: Donchian(20) breakout on 12h with 1-week trend filter and volume confirmation
# Long: Price breaks above Donchian high in weekly uptrend with volume > 20-period average
# Short: Price breaks below Donchian low in weekly downtrend with volume > 20-period average
# Exit: Price crosses Donchian middle or weekly trend reverses
# Position sizing: 0.25 for balance between return and risk
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "12h"
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
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === 12H DONCHIAN CHANNEL ===
    donch_length = 20
    donch_high = pd.Series(high).rolling(window=donch_length, min_periods=donch_length).max().values
    donch_low = pd.Series(low).rolling(window=donch_length, min_periods=donch_length).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        uptrend = close[i] > weekly_ema_aligned[i]
        downtrend = close[i] < weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle OR weekly trend turns down
            if close[i] < donch_mid[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR weekly trend turns up
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