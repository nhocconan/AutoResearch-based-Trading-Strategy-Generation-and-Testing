#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3, 1d EMA34 up-trend, volume > 1.5x average
# Short when price breaks below S3, 1d EMA34 down-trend, volume > 1.5x average
# Exit when price reverts to daily pivot (mean reversion)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag while ensuring sufficient trades.
# Uses 1d for signal direction/trend, 6h only for entry timing and breakout levels.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Camarilla pivot calculation (using prior day's OHLC)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC from 6h data for prior day's pivot
    # We'll use the prior completed day's OHLC to calculate today's Camarilla levels
    df_6h_copy = df_6h.copy()
    df_6h_copy['date'] = pd.to_datetime(df_6h_copy.index).date
    daily_ohlc = df_6h_copy.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using prior day's OHLC
    camarilla_r3 = np.full(len(daily_ohlc), np.nan)
    camarilla_s3 = np.full(len(daily_ohlc), np.nan)
    daily_pivot = np.full(len(daily_ohlc), np.nan)
    
    for i in range(1, len(daily_ohlc)):
        # Prior day's OHLC
        phigh = daily_ohlc.iloc[i-1]['high']
        plow = daily_ohlc.iloc[i-1]['low']
        pclose = daily_ohlc.iloc[i-1]['close']
        
        # Camarilla calculations
        range_val = phigh - plow
        daily_pivot[i] = (phigh + plow + pclose) / 3
        camarilla_r3[i] = daily_pivot[i] + range_val * 1.1 / 4
        camarilla_s3[i] = daily_pivot[i] - range_val * 1.1 / 4
    
    # Forward fill to get today's levels based on prior day's data
    camarilla_r3 = pd.Series(camarilla_r3).ffill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().values
    daily_pivot = pd.Series(daily_pivot).ffill().values
    
    # Create mapping from 6h bar to daily index
    df_6h_copy['date_only'] = pd.to_datetime(df_6h_copy.index).date
    date_to_idx = {date: idx for idx, date in enumerate(daily_ohlc['date'])}
    daily_idx_for_6h = df_6h_copy['date_only'].map(date_to_idx).values
    
    # Get Camarilla levels for each 6h bar
    camarilla_r3_6h = camarilla_r3[daily_idx_for_6h]
    camarilla_s3_6h = camarilla_s3[daily_idx_for_6h]
    daily_pivot_6h = daily_pivot[daily_idx_for_6h]
    
    # Align 6h indicators to 6h timeframe (no additional delay needed as they're based on prior day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3_6h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3_6h)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_6h, daily_pivot_6h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume MA and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(daily_pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_pivot = daily_pivot_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below daily pivot (mean reversion)
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above daily pivot (mean reversion)
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above R3, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals