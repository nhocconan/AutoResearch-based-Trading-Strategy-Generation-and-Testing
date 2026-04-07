#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Donchian breakout for trend direction and 1d EMA for trend filter, with volume confirmation for institutional participation. Enter long when price breaks above 10-period weekly Donchian high with price above EMA and volume confirmation; enter short when price breaks below 10-period weekly Donchian low with price below EMA and volume confirmation. Exit when price returns to midline or opposite breakout occurs. This strategy captures strong trending moves with volume confirmation, reducing false signals and trade frequency. Weekly trend filter works in bull/bear via breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
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
    
    # Weekly data for Donchian and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels on weekly data (10-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high and low (10-period)
    donch_high = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    donch_low = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Weekly EMA for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    
    # Align indicators to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_12h = align_htf_to_ltf(prices, df_1w, donch_mid)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or
            np.isnan(donch_mid_12h[i]) or np.isnan(ema_1w_12h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from weekly EMA
        uptrend = close[i] > ema_1w_12h[i]
        downtrend = close[i] < ema_1w_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to midline (trend weakening)
            if close[i] <= donch_mid_12h[i]:
                exit_long = True
            # Exit if price breaks below weekly Donchian low (strong reversal)
            elif close[i] < donch_low_12h[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and close[i] < donch_high_12h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to midline (trend weakening)
            if close[i] >= donch_mid_12h[i]:
                exit_short = True
            # Exit if price breaks above weekly Donchian high (strong reversal)
            elif close[i] > donch_high_12h[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and close[i] > donch_low_12h[i]:
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
            if close[i] > donch_high_12h[i] and close[i-1] <= donch_high_12h[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below weekly Donchian low with downtrend and volume confirmation
            if close[i] < donch_low_12h[i] and close[i-1] >= donch_low_12h[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals