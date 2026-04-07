#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, use weekly Donchian channel breakout with weekly EMA trend filter and volume confirmation. Enter long when price breaks above weekly Donchian high with price above weekly EMA and volume > 1.5x average; enter short when price breaks below weekly Donchian low with price below weekly EMA and volume confirmation. Exit on opposite Donchian break or trend reversal. This captures strong weekly trends with institutional volume, works in bull/bear via trend filter, and limits trades to avoid fee drag.
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
    
    # Weekly data for Donchian and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channel (20 periods)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA for trend filter (50 periods)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to daily timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_1w_1d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-period average on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or
            np.isnan(ema_1w_1d[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from weekly EMA
        uptrend = close[i] > ema_1w_1d[i]
        downtrend = close[i] < ema_1w_1d[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price breaks below weekly Donchian low
            if close[i] < donchian_low_1d[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price breaks above weekly Donchian high
            if close[i] > donchian_high_1d[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above weekly Donchian high with uptrend and volume confirmation
            if close[i] > donchian_high_1d[i] and close[i-1] <= donchian_high_1d[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below weekly Donchian low with downtrend and volume confirmation
            if close[i] < donchian_low_1d[i] and close[i-1] >= donchian_low_1d[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals