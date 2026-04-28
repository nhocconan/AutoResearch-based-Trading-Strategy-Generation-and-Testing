#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation.
# Uses 4h primary timeframe for lower trade frequency and better generalization.
# Camarilla H3/L3 levels provide structured breakouts with moderate frequency.
# 12h EMA34 filters for trend alignment on higher timeframe, reducing counter-trend trades.
# Volume spike confirms breakout strength and filters low-momentum false breakouts.
# Position size 0.25 for balanced risk/return. Target: 80-180 total trades over 4 years (20-45/year).

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h Camarilla pivots (based on previous day's OHLC)
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
    
    # Camarilla H3, L3 levels
    H3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    L3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    
    # Map daily levels to 4h bars
    date_map = prices_df.set_index('open_time')['date']
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    
    for idx, date_val in enumerate(date_map):
        if pd.isna(date_val):
            continue
        date_idx = daily_ohlc[daily_ohlc['date'] == date_val].index
        if len(date_idx) > 0 and date_idx[0] > 0:  # Ensure we have previous day
            prev_idx = date_idx[0] - 1
            camarilla_H3[idx] = H3[prev_idx]
            camarilla_L3[idx] = L3[prev_idx]
    
    # 4h volume spike: >1.8x 20-bar average volume (stricter for lower TF)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(camarilla_H3[i]) or
            np.isnan(camarilla_L3[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA34 direction
        price_above_ema = close[i] > ema_34_12h_aligned[i]
        price_below_ema = close[i] < ema_34_12h_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > camarilla_H3[i]
        short_breakout = close[i] < camarilla_L3[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_L3[i]
        short_exit = close[i] > camarilla_H3[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals