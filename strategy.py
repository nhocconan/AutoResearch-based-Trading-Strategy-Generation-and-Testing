#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume confirmation.
# Uses 1w EMA200 for robust long-term trend alignment (works in both bull and bear markets).
# Long when price breaks above R3 AND price > 1w EMA200 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND price < 1w EMA200 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# 1d timeframe provides stable structure with lower trade frequency to minimize fee drag.
# Volume confirmation ensures only high-conviction breakouts are traded.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Camarilla_R3S3_Breakout_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA200 trend filter
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
    
    # Calculate daily Camarilla levels from 1d OHLC
    df_1d = prices.copy()
    df_1d['date'] = pd.to_datetime(df_1d['open_time']).dt.date
    daily = df_1d.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    daily['camarilla_r3'] = daily['close'] + (daily['high'] - daily['low']) * 1.1 / 4
    daily['camarilla_s3'] = daily['close'] - (daily['high'] - daily['low']) * 1.1 / 4
    
    # Map daily levels to 1d bars
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily[daily['date'] == date]
        if len(day_row) > 0:
            camarilla_r3[i] = day_row.iloc[0]['camarilla_r3']
            camarilla_s3[i] = day_row.iloc[0]['camarilla_s3']
    
    # Volume confirmation: current 1d volume > 2.0x 20-bar average
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