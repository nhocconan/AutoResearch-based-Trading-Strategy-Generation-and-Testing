#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 1d primary timeframe for lower frequency and higher reliability, 1w for trend direction.
# Camarilla pivots provide precise support/resistance levels based on prior week's range,
# filtered by 1w EMA34 trend and volume spikes to avoid false breakouts. Designed to work
# in both bull and bear markets by following the weekly trend while using Camarilla levels
# as structure. Target: 30-100 total trades over 4 years = 7-25/year for 1d. Size: 0.25.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla pivots (based on previous week's OHLC)
    # Group by week to get weekly OHLC
    prices_df = prices.copy()
    prices_df['week'] = prices_df['open_time'].dt.to_period('W').dt.start_time
    weekly_ohlc = prices_df.groupby('week').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate Camarilla levels for each week
    high_prev = weekly_ohlc['high'].shift(1).values
    low_prev = weekly_ohlc['low'].shift(1).values
    close_prev = weekly_ohlc['close'].shift(1).values
    
    # Camarilla R3, S3 levels
    R3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    S3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    
    # Map weekly levels to 1d bars
    week_map = prices_df.set_index('open_time')['week']
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for idx, week_val in enumerate(week_map):
        if pd.isna(week_val):
            continue
        week_idx = weekly_ohlc[weekly_ohlc['week'] == week_val].index
        if len(week_idx) > 0 and week_idx[0] > 0:  # Ensure we have previous week
            prev_idx = week_idx[0] - 1
            camarilla_R3[idx] = R3[prev_idx]
            camarilla_S3[idx] = S3[prev_idx]
    
    # 1d volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(camarilla_R3[i]) or
            np.isnan(camarilla_S3[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > camarilla_R3[i]
        short_breakout = close[i] < camarilla_S3[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_S3[i]
        short_exit = close[i] > camarilla_R3[i]
        
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