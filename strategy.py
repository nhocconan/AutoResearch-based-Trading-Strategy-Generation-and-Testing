#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points using previous week's data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Weekly pivot point
    weekly_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h RSI(14) for momentum
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_ma = np.full(n, np.nan)
    loss_ma = np.full(n, np.nan)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    gain_ma_values = gain_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    loss_ma_values = loss_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    gain_ma[:] = gain_ma_values
    loss_ma[:] = loss_ma_values
    
    rs = gain_ma / loss_ma
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period EMA
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema[:] = vol_ema_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ema[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x EMA
        volume_filter = volume[i] > vol_ema[i] * 1.3
        
        # Entry conditions: RSI extreme + price near weekly pivot (mean reversion)
        long_entry = (rsi[i] < 30) and (close[i] <= weekly_pivot_6h[i] * 1.02) and volume_filter
        short_entry = (rsi[i] > 70) and (close[i] >= weekly_pivot_6h[i] * 0.98) and volume_filter
        
        # Exit conditions: RSI returns to neutral zone
        long_exit = rsi[i] > 50
        short_exit = rsi[i] < 50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_rsi_pivot_mean_reversion_vol_filter_v1"
timeframe = "6h"
leverage = 1.0