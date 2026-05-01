#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume confirmation.
# Uses 1w EMA200 for robust long-term trend alignment (works in both bull and bear markets).
# Long when price breaks above R3 AND price > 1w EMA200 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND price < 1w EMA200 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# 1w timeframe provides stable trend filter less prone to whipsaw vs shorter HTF.
# Volume confirmation ensures only high-conviction breakouts are traded.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Camarilla_R3S3_Breakout_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Load 1w data ONCE before loop for EMA200 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA200 calculation
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # 1w trend: price above/below EMA200
    price_above_ema = close > ema_200_aligned
    price_below_ema = close < ema_200_aligned
    
    # Calculate Camarilla levels (based on previous 1w bar's range)
    df_1w_copy = df_1w.copy()
    df_1w_copy['date'] = pd.to_datetime(df_1w_copy['open_time']).dt.date
    weekly_1w = df_1w_copy.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1w week
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    weekly_1w['camarilla_r3'] = weekly_1w['close'] + (weekly_1w['high'] - weekly_1w['low']) * 1.1 / 4
    weekly_1w['camarilla_s3'] = weekly_1w['close'] - (weekly_1w['high'] - weekly_1w['low']) * 1.1 / 4
    
    # Map weekly 1w levels to 6h bars
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        week_row = weekly_1w[weekly_1w['date'] == date]
        if len(week_row) > 0:
            camarilla_r3[i] = week_row.iloc[0]['camarilla_r3']
            camarilla_s3[i] = week_row.iloc[0]['camarilla_s3']
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]):
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
            # Long: breakout above R3 AND price > 1w EMA200 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND price < 1w EMA200 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR price < 1w EMA200 (trend change)
            if (curr_low < camarilla_s3[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR price > 1w EMA200 (trend change)
            if (curr_high > camarilla_r3[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals