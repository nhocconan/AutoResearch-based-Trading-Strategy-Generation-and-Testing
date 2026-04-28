#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA50 trend filter and volume confirmation.
# Uses 1h primary timeframe for entry timing, 4h for trend direction and structure.
# Camarilla H4/L4 levels (closer to price) provide more frequent but structured breakouts.
# 4h EMA50 filters for trend alignment, volume spike confirms breakout strength.
# Session filter (08-20 UTC) reduces noise. Position size 0.20 for risk control.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_H4L4_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivots (based on previous day's OHLC)
    # Group by date to get daily OHLC
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    daily_ohlc = prices_df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate Camarilla levels for each day
    high_prev = daily_ohlc['high'].shift(1).values
    low_prev = daily_ohlc['low'].shift(1).values
    close_prev = daily_ohlc['close'].shift(1).values
    
    # Camarilla H4, L4 levels (closer to price)
    H4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    L4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Map daily levels to 1h bars
    date_map = prices_df.set_index('open_time')['date']
    camarilla_H4 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    
    for idx, date_val in enumerate(date_map):
        if pd.isna(date_val):
            continue
        date_idx = daily_ohlc[daily_ohlc['date'] == date_val].index
        if len(date_idx) > 0 and date_idx[0] > 0:  # Ensure we have previous day
            prev_idx = date_idx[0] - 1
            camarilla_H4[idx] = H4[prev_idx]
            camarilla_L4[idx] = L4[prev_idx]
    
    # 1h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(camarilla_H4[i]) or
            np.isnan(camarilla_L4[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 direction
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > camarilla_H4[i]
        short_breakout = close[i] < camarilla_L4[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_L4[i]
        short_exit = close[i] > camarilla_H4[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals