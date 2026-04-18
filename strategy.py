#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Get daily data for Donchian breakout
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-day) based on previous day's data
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    lookback = 20
    if len(high_1d) >= lookback:
        for i in range(lookback, len(high_1d)):
            upper[i] = np.max(high_1d[i-lookback:i])
            lower[i] = np.min(low_1d[i-lookback:i])
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align all data to daily timeframe
    upper_daily = align_htf_to_ltf(prices, df_1d, upper)
    lower_daily = align_htf_to_ltf(prices, df_1d, lower)
    vol_ma_daily = align_htf_to_ltf(prices, df_1d, vol_ma)
    ema_1w_daily = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_daily[i]) or np.isnan(lower_daily[i]) or 
            np.isnan(vol_ma_daily[i]) or np.isnan(ema_1w_daily[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma_daily[i]
        
        # Trend filter: price above weekly EMA (bullish bias)
        bullish_bias = close[i] > ema_1w_daily[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume in bullish bias
            if close[i] > upper_daily[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR price falls below weekly EMA
            if close[i] < lower_daily[i] or close[i] < ema_1w_daily[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: price breaks below lower Donchian with volume in bearish bias
            if close[i] < lower_daily[i] and vol_confirm and not bullish_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0  # flat when conditions not met
    
    return signals

name = "1d_Donchian_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0