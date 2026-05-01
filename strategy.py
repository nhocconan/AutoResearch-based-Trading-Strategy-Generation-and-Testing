#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d ADX regime filter.
# Uses 1d ADX > 25 to identify strong trends, reducing whipsaws in ranging markets.
# 4h volume > 2.0x 20-bar average confirms institutional participation.
# Long when price breaks above R3 AND 1d ADX > 25 AND 4h volume spike.
# Short when price breaks below S3 AND 1d ADX > 25 AND 4h volume spike.
# Uses discrete sizing 0.20 to manage drawdown and reduce fee churn.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
# Session filter: 08-20 UTC to avoid low-liquidity Asian session noise.

name = "1h_Camarilla_R3S3_Breakout_4hVolSpike_1dADX25_Trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing function
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
    
    # Align 1d ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for volume MA
        return np.zeros(n)
    
    # 4h volume MA
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate Camarilla levels (based on previous 1d bar's range)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_r3_1d = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3_1d = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    start_idx = 50  # warmup for ADX and volume MA
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        # Volume confirmation: current 1h volume > 2.0x 4h volume MA (scaled)
        # Scale 4h MA to 1h equivalent: 4h MA represents 4 bars, so divide by 4 for per-hour baseline
        vol_ma_1h_baseline = vol_ma_4h_aligned[i] / 4.0
        if vol_ma_1h_baseline <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (vol_ma_1h_baseline * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND 1d ADX > 25 AND volume confirmation
            if (breakout_up and 
                adx_aligned[i] > 25 and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND 1d ADX > 25 AND volume confirmation
            elif (breakout_down and 
                  adx_aligned[i] > 25 and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR ADX < 20 (trend weakening)
            if (curr_low < camarilla_s3_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR ADX < 20 (trend weakening)
            if (curr_high > camarilla_r3_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals