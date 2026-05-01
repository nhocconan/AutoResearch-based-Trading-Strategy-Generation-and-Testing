#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian channel AND 1w EMA50 rising AND volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian channel AND 1w EMA50 falling AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture medium-term trends with low trade frequency.
# Donchian channels provide robust price structure that works in both trending and ranging markets.
# 1w EMA50 trend filter ensures alignment with higher timeframe momentum to avoid counter-trend trades.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and improve test generalization.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 1w EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Calculate Donchian channels (20-period) from 1d data
    # Upper channel: highest high over past 20 periods
    # Lower channel: lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 1d volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian, EMA and volume MA calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 1d timeframe
        hour = hours[i]
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper channel
        breakout_down = curr_low < donchian_low[i]  # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper channel AND 1w EMA50 rising AND volume confirmation
            if (breakout_up and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel AND 1w EMA50 falling AND volume confirmation
            elif (breakout_down and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower Donchian channel (stoploss) OR 1w EMA50 falls (trend change)
            if (curr_low < donchian_low[i] or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian channel (stoploss) OR 1w EMA50 rises (trend change)
            if (curr_high > donchian_high[i] or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals