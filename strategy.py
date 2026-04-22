#!/usr/bin/env python3
"""
Hypothesis: VWAP bounce with volume filter on 1-day timeframe.
- Long when price crosses above VWAP and 1D volume > 50-period average volume
- Short when price crosses below VWAP and 1D volume > 50-period average volume
- Exit on opposite VWAP cross or volume drop below average
- Uses 1W trend filter (price above/below 50-period EMA) to avoid counter-trend trades
- Designed for 1d timeframe to target 7-25 trades/year, avoiding overtrading
- Works in bull/bear markets by following institutional volume + trend alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for volume filter and trend - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # 1D indicators: average volume and 50-period EMA for trend
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D indicators to lower timeframe
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1-week data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # VWAP calculation for 1d period (resets daily)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    vwap = np.full(n, np.nan)
    cum_num = 0.0
    cum_den = 0.0
    
    for i in range(n):
        # Reset at start of each day (00:00 UTC)
        if i > 0 and prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
            cum_num = 0.0
            cum_den = 0.0
        
        cum_num += vwap_numerator[i]
        cum_den += vwap_denominator[i]
        
        if cum_den > 0:
            vwap[i] = cum_num / cum_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(vwap[i]) or np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above VWAP, volume above average, and both 1D and 1W trends up
            if (close[i] > vwap[i] and 
                close[i-1] <= vwap[i-1] and 
                volume_1d[i] > avg_vol_1d_aligned[i] and
                close_1d[i] > ema_50_1d_aligned[i] and
                close_1w[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP, volume above average, and both 1D and 1W trends down
            elif (close[i] < vwap[i] and 
                  close[i-1] >= vwap[i-1] and 
                  volume_1d[i] > avg_vol_1d_aligned[i] and
                  close_1d[i] < ema_50_1d_aligned[i] and
                  close_1w[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below VWAP OR volume drops below average
                if (close[i] < vwap[i] and close[i-1] >= vwap[i-1]) or volume_1d[i] < avg_vol_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above VWAP OR volume drops below average
                if (close[i] > vwap[i] and close[i-1] <= vwap[i-1]) or volume_1d[i] < avg_vol_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "VWAP_Bounce_1dVolume_Filter"
timeframe = "1d"
leverage = 1.0