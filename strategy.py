#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 level AND 1d EMA50 rising AND volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S4 level AND 1d EMA50 falling AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Camarilla R4/S4 levels provide stronger breakout signals
# than R3/S3, reducing false breakouts and trade frequency. 1d EMA50 ensures alignment with daily trend.
# Volume confirmation filters low-momentum breakouts. Designed for 4h timeframe to capture medium-term swings
# with controlled trade frequency in both bull and bear markets.

name = "4h_Camarilla_R4S4_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Camarilla levels calculation (based on prior day's OHLC)
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    # Shift by 1 to get prior day's data
    prior_day = prices_df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)
    
    # Map prior day's OHLC back to each 4h bar
    prior_high = prices_df['date'].map(prior_day['high']).values
    prior_low = prices_df['date'].map(prior_day['low']).values
    prior_close = prices_df['date'].map(prior_day['close']).values
    
    # Calculate Camarilla levels
    rang = prior_high - prior_low
    camarilla_r4 = prior_close + rang * 1.1 / 2
    camarilla_s4 = prior_close - rang * 1.1 / 2
    
    # Volume confirmation: current 4h volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Camarilla calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r4[i]  # break above R4 level
        breakout_down = curr_low < camarilla_s4[i]  # break below S4 level
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R4 AND 1d EMA50 rising AND volume confirmation
            if (breakout_up and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Camarilla S4 AND 1d EMA50 falling AND volume confirmation
            elif (breakout_down and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S4 (stoploss) OR 1d EMA50 falls (trend change)
            if (curr_low < camarilla_s4[i] or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R4 (stoploss) OR 1d EMA50 rises (trend change)
            if (curr_high > camarilla_r4[i] or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals