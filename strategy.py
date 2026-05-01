#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R4 AND 1d EMA34 rising AND volume > 1.8x 24-bar average.
# Short when price breaks below S4 AND 1d EMA34 falling AND volume > 1.8x 24-bar average.
# Uses discrete sizing 0.20 to minimize fee churn. Session filter 08-20 UTC.
# Target: 60-120 total trades over 4 years (15-30/year) for BTC/ETH/SOL.
# R4/S4 levels are stronger breakout levels than R3/S3, reducing false signals.
# 1d EMA34 provides robust trend filter aligned with daily momentum.
# Higher volume threshold (1.8x) ensures only significant breakouts are traded.

name = "1h_Camarilla_R4S4_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
    # Calculate Camarilla levels (based on previous day's range)
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla R4 = close + (high - low) * 1.1/2
    # Camarilla S4 = close - (high - low) * 1.1/2
    daily['camarilla_r4'] = daily['close'] + (daily['high'] - daily['low']) * 1.1 / 2
    daily['camarilla_s4'] = daily['close'] - (daily['high'] - daily['low']) * 1.1 / 2
    
    # Map daily levels to 1h bars
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily[daily['date'] == date]
        if len(day_row) > 0:
            camarilla_r4[i] = day_row.iloc[0]['camarilla_r4']
            camarilla_s4[i] = day_row.iloc[0]['camarilla_s4']
    
    # Volume confirmation: current 1h volume > 1.8x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r4[i]  # break above R4
        breakout_down = curr_low < camarilla_s4[i]  # break below S4
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R4 AND 1d EMA34 rising AND volume confirmation
            if (breakout_up and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S4 AND 1d EMA34 falling AND volume confirmation
            elif (breakout_down and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S4 (stoploss) OR 1d EMA34 falls (trend change)
            if (curr_low < camarilla_s4[i] or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R4 (stoploss) OR 1d EMA34 rises (trend change)
            if (curr_high > camarilla_r4[i] or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals