#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA50 for long-term trend alignment to reduce whipsaws in ranging markets.
# Long when price breaks above R4 AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S4 AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 00-24 UTC (always on).
# Volume spike threshold set to 2.0x to reduce overtrading and improve signal quality.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R4S4_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Pre-compute session hours for efficiency (always on for 12h)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Calculate Camarilla levels (based on previous 1d bar's range)
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
    # Camarilla R4 = close + (high - low) * 1.1/2
    # Camarilla S4 = close - (high - low) * 1.1/2
    daily_1d['camarilla_r4'] = daily_1d['close'] + (daily_1d['high'] - daily_1d['low']) * 1.1 / 2
    daily_1d['camarilla_s4'] = daily_1d['close'] - (daily_1d['high'] - daily_1d['low']) * 1.1 / 2
    
    # Map daily 1d levels to 12h bars
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily_1d[daily_1d['date'] == date]
        if len(day_row) > 0:
            camarilla_r4[i] = day_row.iloc[0]['camarilla_r4']
            camarilla_s4[i] = day_row.iloc[0]['camarilla_s4']
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: always on for 12h (no restriction)
        hour = hours[i]
        # No session filter for 12h - trade all hours
        
        if np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > camarilla_r4[i]  # break above R4
        breakout_down = curr_low < camarilla_s4[i]  # break below S4
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R4 AND price > 1d EMA50 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S4 AND price < 1d EMA50 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S4 (stoploss) OR price < 1d EMA50 (trend change)
            if (curr_low < camarilla_s4[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R4 (stoploss) OR price > 1d EMA50 (trend change)
            if (curr_high > camarilla_r4[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals