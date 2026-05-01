#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R1 level AND 4h EMA50 rising AND volume > 2.0x 24-bar average.
# Short when price breaks below Camarilla S1 level AND 4h EMA50 falling AND volume > 2.0x 24-bar average.
# Uses discrete sizing 0.20 to minimize fee churn. Designed for 1h timeframe to capture medium-term trends with low trade frequency.
# Camarilla levels derived from prior day's range, providing institutional pivot points that work in both bull and bear markets.
# 4h EMA50 trend filter ensures alignment with higher timeframe momentum.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Session filter (08-20 UTC) reduces noise trades during low liquidity periods.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 calculation
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 4h EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Camarilla levels calculation (based on prior day's OHLC)
    # Need to group by day to get prior day's OHLC
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    # Shift by 1 to get prior day's data
    prior_day = prices_df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)
    
    # Map prior day's OHLC back to each 1h bar
    prior_high = prices_df['date'].map(prior_day['high']).values
    prior_low = prices_df['date'].map(prior_day['low']).values
    prior_close = prices_df['date'].map(prior_day['close']).values
    
    # Calculate Camarilla levels
    rang = prior_high - prior_low
    camarilla_r1 = prior_close + rang * 1.1 / 12
    camarilla_s1 = prior_close - rang * 1.1 / 12
    
    # Volume confirmation: current 1h volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Camarilla calculation
    
    for i in range(start_idx, n):
        # Session filter: trade only during 08-20 UTC
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r1[i]  # break above R1 level
        breakout_down = curr_low < camarilla_s1[i]  # break below S1 level
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R1 AND 4h EMA50 rising AND volume confirmation
            if (breakout_up and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below Camarilla S1 AND 4h EMA50 falling AND volume confirmation
            elif (breakout_down and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S1 (stoploss) OR 4h EMA50 falls (trend change)
            if (curr_low < camarilla_s1[i] or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R1 (stoploss) OR 4h EMA50 rises (trend change)
            if (curr_high > camarilla_r1[i] or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals