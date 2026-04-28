#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 12h primary timeframe for lower trade frequency, 1d for trend direction and structure.
# Camarilla H3/L3 levels provide structured breakouts with good risk/reward.
# 1d EMA34 filters for trend alignment on higher timeframe, volume spike confirms breakout strength.
# Position size 0.25 for risk control. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla pivots (based on previous day's OHLC)
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
    
    # Map daily levels to 12h bars
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
    
    # 12h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_H3[i]) or
            np.isnan(camarilla_L3[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
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