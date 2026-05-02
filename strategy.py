#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 Trend Filter and Volume Spike
# Uses 4h EMA50 for trend direction and 1h Camarilla levels for precise entry timing.
# Volume spike confirms breakout validity. Designed for 1h timeframe to capture
# intraday trends while using 4h trend filter to avoid counter-trend trades.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Works in both bull and bear markets by only taking trades in direction of 4h trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels (using previous day's OHLC)
    # Group by date to get daily OHLC
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    daily_ohlc = prices_df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Shift to use previous day's levels (no look-ahead)
    daily_ohlc['prev_high'] = daily_ohlc['high'].shift(1)
    daily_ohlc['prev_low'] = daily_ohlc['low'].shift(1)
    daily_ohlc['prev_close'] = daily_ohlc['close'].shift(1)
    daily_ohlc = daily_ohlc.dropna()
    
    if len(daily_ohlc) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    daily_ohlc['camarilla_r3'] = daily_ohlc['prev_close'] + 1.1 * (daily_ohlc['prev_high'] - daily_ohlc['prev_low']) / 12
    daily_ohlc['camarilla_s3'] = daily_ohlc['prev_close'] - 1.1 * (daily_ohlc['prev_high'] - daily_ohlc['prev_low']) / 12
    daily_ohlc['camarilla_r4'] = daily_ohlc['prev_close'] + 1.1 * (daily_ohlc['prev_high'] - daily_ohlc['prev_low']) / 6
    daily_ohlc['camarilla_s4'] = daily_ohlc['prev_close'] - 1.1 * (daily_ohlc['prev_high'] - daily_ohlc['prev_low']) / 6
    
    # Map daily levels to 1h bars (forward fill)
    camarilla_data = daily_ohlc[['date', 'camarilla_r3', 'camarilla_s3', 'camarilla_r4', 'camarilla_s4']]
    camarilla_data = camarilla_data.set_index('date')
    
    prices_df = prices_df.set_index('date')
    prices_df = prices_df.join(camarilla_data, how='left')
    prices_df = prices_df.reset_index(drop=True)
    
    camarilla_r3 = prices_df['camarilla_r3'].ffill().values
    camarilla_s3 = prices_df['camarilla_s3'].ffill().values
    camarilla_r4 = prices_df['camarilla_r4'].ffill().values
    camarilla_s4 = prices_df['camarilla_s4'].ffill().values
    
    # Volume confirmation: 2.0x 24-period average (1 day for 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and Camarilla)
    start_idx = max(50, 24)  # 4h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 4h EMA50 (bullish trend)
            if (close[i] > camarilla_r3[i] and 
                volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 4h EMA50 (bearish trend)
            elif (close[i] < camarilla_s3[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_s3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_r3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals