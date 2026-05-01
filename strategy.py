#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (price > EMA34) and volume confirmation.
# Long when price breaks above R3 AND price > 1d EMA34 AND volume > 1.5x 20-bar average.
# Short when price breaks below S3 AND price < 1d EMA34 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to balance drawdown and trade frequency.
# Target: 80-150 total trades over 4 years (20-37/year) for 4h timeframe.
# 1d EMA34 provides smooth trend alignment that works in both bull and bear markets.
# Camarilla R3/S3 levels offer reliable breakout points with lower noise.
# Volume confirmation ensures only high-conviction breakouts are traded.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels (based on previous day's range)
    df_1d_copy = df_1d.copy()
    df_1d_copy['date'] = pd.to_datetime(df_1d_copy['open_time']).dt.date
    daily = df_1d_copy.groupby('date').agg({
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
    
    # Map daily levels to 4h bars
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily[daily['date'] == date]
        if len(day_row) > 0:
            camarilla_r3[i] = day_row.iloc[0]['camarilla_r3']
            camarilla_s3[i] = day_row.iloc[0]['camarilla_s3']
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend flags: price above/below 1d EMA34
    price_above_ema = close > ema_34_aligned
    price_below_ema = close < ema_34_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3[i]  # break above R3
        breakout_down = curr_low < camarilla_s3[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND price > 1d EMA34 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND price < 1d EMA34 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR price < 1d EMA34 (trend change)
            if (curr_low < camarilla_s3[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR price > 1d EMA34 (trend change)
            if (curr_high > camarilla_r3[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals