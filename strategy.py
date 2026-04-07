#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, use weekly Donchian channels for breakout direction, with weekly EMA for trend filter, and volume confirmation for institutional participation. Enter long when price breaks above upper Donchian with price above weekly EMA and volume confirmation; enter short when price breaks below lower Donchian with price below weekly EMA and volume confirmation. Exit when price returns to the Donchian midline or opposite side. This strategy targets strong trending moves with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via trend filter and breakout logic. Target: 30-100 trades over 4 years.
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
    
    # Calculate Donchian Channel (20-period) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Upper band: highest high of last 20 weeks
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weeks
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Middle line: average of upper and lower
    middle_20 = (upper_20 + lower_20) / 2
    
    # Weekly EMA for trend filter (50-period)
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to daily timeframe
    upper_20_d = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_d = align_htf_to_ltf(prices, df_1w, lower_20)
    middle_20_d = align_htf_to_ltf(prices, df_1w, middle_20)
    ema_1w_d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-period average on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_20_d[i]) or np.isnan(lower_20_d[i]) or
            np.isnan(middle_20_d[i]) or np.isnan(ema_1w_d[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from weekly EMA
        uptrend = close[i] > ema_1w_d[i]
        downtrend = close[i] < ema_1w_d[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to or below middle line
            if close[i] <= middle_20_d[i]:
                exit_long = True
            # Exit if price breaks below lower band (reversal)
            elif close[i] < lower_20_d[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and close[i] < ema_1w_d[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to or above middle line
            if close[i] >= middle_20_d[i]:
                exit_short = True
            # Exit if price breaks above upper band (reversal)
            elif close[i] > upper_20_d[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and close[i] > ema_1w_d[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above upper Donchian with uptrend and volume confirmation
            if close[i] > upper_20_d[i] and close[i-1] <= upper_20_d[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below lower Donchian with downtrend and volume confirmation
            if close[i] < lower_20_d[i] and close[i-1] >= lower_20_d[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals