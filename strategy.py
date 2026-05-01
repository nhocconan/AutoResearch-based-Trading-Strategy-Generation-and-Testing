#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level AND 4h EMA34 rising AND volume > 2.0x 24-bar average.
# Short when price breaks below Camarilla S3 level AND 4h EMA34 falling AND volume > 2.0x 24-bar average.
# Uses discrete sizing 0.20 to minimize fee churn. Session filter (08-20 UTC) reduces noise trades.
# Designed for 1h timeframe with 4h/1d HTF for signal direction, targeting 15-37 trades/year.
# Camarilla levels derived from prior day's range, providing institutional pivot points that work in both bull and bear markets.
# 4h EMA34 trend filter ensures alignment with higher timeframe momentum.
# Volume spike requirement reduces false breakouts and improves signal quality.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 calculation
    close_4h = df_4h['close'].values
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # 4h EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
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
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    
    # Volume confirmation: current 1h volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Camarilla calculation
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3[i]  # break above R3 level
        breakout_down = curr_low < camarilla_s3[i]  # break below S3 level
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R3 AND 4h EMA34 rising AND volume confirmation
            if (breakout_up and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below Camarilla S3 AND 4h EMA34 falling AND volume confirmation
            elif (breakout_down and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S3 (stoploss) OR 4h EMA34 falls (trend change)
            if (curr_low < camarilla_s3[i] or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R3 (stoploss) OR 4h EMA34 rises (trend change)
            if (curr_high > camarilla_r3[i] or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals