#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX trend filter and volume confirmation.
# Uses 4h ADX > 25 to identify strong trends, reducing whipsaws in ranging markets.
# Long when price breaks above R3 AND 4h ADX > 25 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND 4h ADX > 25 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.20 to manage drawdown. Session filter 08-20 UTC to reduce noise.
# Volume spike threshold set to 2.0x to ensure high-quality signals and limit trade frequency.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hADX25_Trend_VolumeSpike_v1"
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
    
    # Load 4h data ONCE before loop for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 4h ADX calculation (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align 4h ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 4h trend: ADX > 25 indicates strong trend
    strong_trend = adx_aligned > 25
    
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
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    daily_1d['camarilla_r3'] = daily_1d['close'] + (daily_1d['high'] - daily_1d['low']) * 1.1 / 4
    daily_1d['camarilla_s3'] = daily_1d['close'] - (daily_1d['high'] - daily_1d['low']) * 1.1 / 4
    
    # Map daily 1d levels to 1h bars
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        date = prices.iloc[i]['open_time'].date()
        day_row = daily_1d[daily_1d['date'] == date]
        if len(day_row) > 0:
            camarilla_r3[i] = day_row.iloc[0]['camarilla_r3']
            camarilla_s3[i] = day_row.iloc[0]['camarilla_s3']
    
    # Volume confirmation: current 1h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ADX and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > camarilla_r3[i]  # break above R3
        breakout_down = curr_low < camarilla_s3[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND 4h ADX > 25 AND volume confirmation
            if (breakout_up and 
                strong_trend[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND 4h ADX > 25 AND volume confirmation
            elif (breakout_down and 
                  strong_trend[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR ADX < 20 (trend weakening)
            if (curr_low < camarilla_s3[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR ADX < 20 (trend weakening)
            if (curr_high > camarilla_r3[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals