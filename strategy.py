#!/usr/bin/env python3
"""
4h_price_channel_breakout_1d_volume_v1
Hypothesis: On 4-hour timeframe, use price channel breakout (Donchian 20) combined with volume confirmation and daily trend filter.
Enter long when price breaks above Donchian upper band AND daily close > daily SMA 50 AND volume > 1.5x 20-period average.
Enter short when price breaks below Donchian lower band AND daily close < daily SMA 50 AND volume > 1.5x 20-period average.
Exit when price crosses opposite Donchian band or volume drops below average.
This strategy targets medium-term trends with volume confirmation to avoid false breakouts, generating ~20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_breakout_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    
    # Calculate daily SMA 50 for trend filter
    d_sma50 = pd.Series(d_close).rolling(window=50, min_periods=50).mean().values
    
    # Align daily SMA to 4h timeframe
    d_sma50_aligned = align_htf_to_ltf(prices, df_1d, d_sma50)
    
    # Calculate Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start from 20 for Donchian
        # Skip if any data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(d_sma50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: daily close vs SMA 50
        daily_uptrend = d_close[min(i//6, len(d_close)-1)] > d_sma50[min(i//6, len(d_sma50)-1)] if i//6 < len(d_close) else False
        daily_downtrend = d_close[min(i//6, len(d_close)-1)] < d_sma50[min(i//6, len(d_sma50)-1)] if i//6 < len(d_close) else False
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses below Donchian lower band
            if low[i] < donchian_low[i]:
                exit_long = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses above Donchian upper band
            if high[i] > donchian_high[i]:
                exit_short = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper AND daily uptrend AND volume confirmed
            long_entry = (high[i] > donchian_high[i]) and daily_uptrend and vol_confirmed
            
            # Short entry: price breaks below Donchian lower AND daily downtrend AND volume confirmed
            short_entry = (low[i] < donchian_low[i]) and daily_downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals