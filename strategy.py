#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit when price touches the opposite Camarilla level (R3/S3) or reverses EMA34 direction.
Uses 1d HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
"""

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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for exit
    camarilla_s4 = np.full(n, np.nan)  # for exit
    
    # Group by date to get daily OHLC for Camarilla calculation
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    daily_ohlc = prices_df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    high_daily = daily_ohlc['high'].values
    low_daily = daily_ohlc['low'].values
    close_daily = daily_ohlc['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3_daily = []
    camarilla_s3_daily = []
    camarilla_r4_daily = []
    camarilla_s4_daily = []
    
    for i in range(len(high_daily)):
        daily_range = high_daily[i] - low_daily[i]
        camarilla_r3_daily.append(close_daily[i] + daily_range * 1.1 / 4)
        camarilla_s3_daily.append(close_daily[i] - daily_range * 1.1 / 4)
        camarilla_r4_daily.append(close_daily[i] + daily_range * 1.1 / 2)
        camarilla_s4_daily.append(close_daily[i] - daily_range * 1.1 / 2)
    
    # Map daily levels to 4h bars
    date_to_idx = {date: idx for idx, date in enumerate(daily_ohlc['date'])}
    for i in range(n):
        bar_date = prices_df.iloc[i]['date']
        if bar_date in date_to_idx:
            idx = date_to_idx[bar_date]
            camarilla_r3[i] = camarilla_r3_daily[idx]
            camarilla_s3[i] = camarilla_s3_daily[idx]
            camarilla_r4[i] = camarilla_r4_daily[idx]
            camarilla_s4[i] = camarilla_s4_daily[idx]
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA (20), EMA34 (34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume spike
            if price > r3 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume spike
            elif price < s3 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S3 OR EMA34 starts falling
                if price <= s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 OR EMA34 starts rising
                if price >= r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0