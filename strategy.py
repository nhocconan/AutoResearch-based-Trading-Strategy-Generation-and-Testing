#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA50 for strong trend alignment to reduce whipsaws in both bull and bear markets.
# Long when price breaks above R1 AND price > 12h EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S1 AND price < 12h EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to reduce noise.
# Volume spike threshold set to 2.0x to ensure high-quality signals and limit trades to target range.
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Calculate Camarilla levels (based on previous 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    df_1d_copy = df_1d.copy()
    df_1d_copy['date'] = pd.to_datetime(df_1d_copy['open_time']).dt.date
    daily_1d = df_1d_copy.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d day
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    daily_1d['camarilla_r1'] = daily_1d['close'] + (daily_1d['high'] - daily_1d['low']) * 1.1 / 12
    daily_1d['camarilla_s1'] = daily_1d['close'] - (daily_1d['high'] - daily_1d['low']) * 1.1 / 12
    
    # Map daily 1d levels to 4h bars
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily_1d[daily_1d['date'] == date]
        if len(day_row) > 0:
            camarilla_r1[i] = day_row.iloc[0]['camarilla_r1']
            camarilla_s1[i] = day_row.iloc[0]['camarilla_s1']
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
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
        
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r1[i]  # break above R1
        breakout_down = curr_low < camarilla_s1[i]  # break below S1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R1 AND price > 12h EMA50 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND price < 12h EMA50 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S1 (stoploss) OR price < 12h EMA50 (trend change)
            if (curr_low < camarilla_s1[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (stoploss) OR price > 12h EMA50 (trend change)
            if (curr_high > camarilla_r1[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals