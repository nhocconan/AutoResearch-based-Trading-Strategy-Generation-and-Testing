#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Uses 4h EMA50 for robust trend alignment (works in both bull and bear markets).
# Long when price breaks above R3 AND price > 4h EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND price < 4h EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.20 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# 4h timeframe reduces noise vs 1d while providing stable trend filter.
# Volume confirmation ensures only high-conviction breakouts are traded.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 calculation
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 4h trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Calculate Camarilla levels (based on previous 4h bar's range)
    df_4h_copy = df_4h.copy()
    df_4h_copy['date'] = pd.to_datetime(df_4h_copy['open_time']).dt.date
    daily_4h = df_4h_copy.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h day
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    daily_4h['camarilla_r3'] = daily_4h['close'] + (daily_4h['high'] - daily_4h['low']) * 1.1 / 4
    daily_4h['camarilla_s3'] = daily_4h['close'] - (daily_4h['high'] - daily_4h['low']) * 1.1 / 4
    
    # Map daily 4h levels to 1h bars
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily_4h[daily_4h['date'] == date]
        if len(day_row) > 0:
            camarilla_r3[i] = day_row.iloc[0]['camarilla_r3']
            camarilla_s3[i] = day_row.iloc[0]['camarilla_s3']
    
    # Volume confirmation: current 1h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3[i]  # break above R3
        breakout_down = curr_low < camarilla_s3[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND price > 4h EMA50 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND price < 4h EMA50 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR price < 4h EMA50 (trend change)
            if (curr_low < camarilla_s3[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR price > 4h EMA50 (trend change)
            if (curr_high > camarilla_r3[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals