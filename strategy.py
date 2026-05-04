#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3 with 1d EMA34 uptrend and volume confirmation
# Short when price breaks below Camarilla S3 with 1d EMA34 downtrend and volume confirmation
# Uses proven Camarilla pivot structure with tight entries to avoid overtrading.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # needed for Camarilla calculation
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    # Camarilla: based on previous day's range
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    # We need previous day's OHLC for each 4h bar
    
    # Convert open_time to date for grouping
    dates = pd.to_datetime(prices['open_time']).dt.date
    unique_dates = pd.Series(dates).drop_duplicates()
    
    # Create arrays for previous day's OHLC
    prev_open = np.full(n, np.nan)
    prev_high = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    prev_close = np.full(n, np.nan)
    
    # For each bar, find the previous day's OHLC
    for i in range(n):
        current_date = dates[i]
        # Find index of current date in unique_dates
        date_idx = np.where(unique_dates == current_date)[0]
        if len(date_idx) > 0 and date_idx[0] > 0:
            prev_date = unique_dates[date_idx[0] - 1]
            # Find first bar of previous date
            prev_date_bars = np.where(dates == prev_date)[0]
            if len(prev_date_bars) > 0:
                first_bar_idx = prev_date_bars[0]
                prev_open[i] = open_price[first_bar_idx]
                prev_high[i] = high[first_bar_idx:first_bar_idx + len(prev_date_bars)].max()
                prev_low[i] = low[first_bar_idx:first_bar_idx + len(prev_date_bars)].min()
                prev_close[i] = close[first_bar_idx:first_bar_idx + len(prev_date_bars)].iloc[-1] if hasattr(close[first_bar_idx:first_bar_idx + len(prev_date_bars)], 'iloc') else close[first_bar_idx:first_bar_idx + len(prev_date_bars)][-1]
    
    # Calculate Camarilla levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])):
            range_val = prev_high[i] - prev_low[i]
            camarilla_r3[i] = prev_close[i] + (range_val * 1.1 / 4)
            camarilla_s3[i] = prev_close[i] - (range_val * 1.1 / 4)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1d trend turns down
            if (close[i] < camarilla_s3[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1d trend turns up
            if (close[i] > camarilla_r3[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals