#!/usr/bin/env python3
"""
4h_donchian_20_1d_volume_v2
Hypothesis: On 4h timeframe, use Donchian channel breakout (20-period) with daily volume confirmation and 1-day EMA50 trend filter. 
Long when price breaks above upper Donchian band with EMA50 uptrend and volume > 1.5x 20-period average. 
Short when price breaks below lower Donchian band with EMA50 downtrend and volume > 1.5x 20-period average.
Exit on opposite Donchian band touch or trend reversal. Targets 20-40 trades/year to minimize fee drag.
Works in bull (breakouts) and bear (breakdowns) by capturing directional moves with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_volume_v2"
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
    
    # Calculate daily EMA50 for trend filter
    daily_close_series = pd.Series(d_close)
    ema50 = daily_close_series.ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_high = donchian_high.values
    donchian_low = donchian_low.values
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if daily EMA not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if Donchian bands not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price touches lower Donchian band
            if close[i] <= donchian_low[i]:
                exit_long = True
            # Exit when trend turns down
            elif not uptrend:
                exit_long = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price touches upper Donchian band
            if close[i] >= donchian_high[i]:
                exit_short = True
            # Exit when trend turns up
            elif not downtrend:
                exit_short = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above upper Donchian band with uptrend and volume
            long_entry = (close[i] > donchian_high[i]) and uptrend and vol_confirmed
            # Short entry: breakdown below lower Donchian band with downtrend and volume
            short_entry = (close[i] < donchian_low[i]) and downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals