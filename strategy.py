#!/usr/bin/env python3
"""
6h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 6h timeframe, use 1d Donchian breakout (20-period) for trend direction and breakout signals, with 1d EMA for trend filter and volume confirmation for institutional participation. Enter long when price breaks above Donchian upper band with price above EMA and volume confirmation; enter short when price breaks below Donchian lower band with price below EMA and volume confirmation. Exit when price returns to the Donchian midpoint or opposite band. This strategy captures strong trending moves with volume confirmation, reducing false signals. Works in bull/bear via trend filter and breakout logic, targeting 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # 1d data for Donchian, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # 1d volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_6h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or
            np.isnan(donchian_mid_6h[i]) or np.isnan(ema_1d_6h[i]) or
            np.isnan(vol_ma_1d_6h[i]) or vol_ma_1d_6h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average on 1d
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_6h[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_1d_6h[i]
        downtrend = close[i] < ema_1d_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to Donchian midpoint (trend weakening)
            if close[i] <= donchian_mid_6h[i]:
                exit_long = True
            # Exit if price breaks below Donchian low (strong reversal)
            elif close[i] < donchian_low_6h[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and close[i] < donchian_mid_6h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to Donchian midpoint (trend weakening)
            if close[i] >= donchian_mid_6h[i]:
                exit_short = True
            # Exit if price breaks above Donchian high (strong reversal)
            elif close[i] > donchian_high_6h[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and close[i] > donchian_mid_6h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above Donchian high with uptrend and volume confirmation
            if close[i] > donchian_high_6h[i] and close[i-1] <= donchian_high_6h[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below Donchian low with downtrend and volume confirmation
            if close[i] < donchian_low_6h[i] and close[i-1] >= donchian_low_6h[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals