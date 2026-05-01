#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 AND 1d EMA34 rising AND volume > 2x 20-bar average.
# Short when price breaks below S3 AND 1d EMA34 falling AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to balance trade frequency and edge.
# Camarilla pivots provide mathematically derived support/resistance levels that work well in crypto.
# 1d EMA34 trend filter ensures alignment with daily momentum.
# Volume spike requirement (2x average) reduces false breakouts and improves signal quality.
# Target: 75-200 total trades over 4 years (19-50/year) for BTC/ETH/SOL.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
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
    
    # Calculate Camarilla levels for each 4h bar using prior 1d OHLC
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need prior 1d close, high, low for each 4h bar
    close_1d_shifted = np.roll(close_1d, 1)  # previous day close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align prior 1d OHLC to 4h timeframe
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = close_1d_aligned + (high_1d_aligned - low_1d_aligned) * 1.1 / 2
    camarilla_s3 = close_1d_aligned - (high_1d_aligned - low_1d_aligned) * 1.1 / 2
    
    # Volume confirmation: current 4h volume > 2x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicator calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > camarilla_r3[i]  # break above R3
        breakout_down = curr_low < camarilla_s3[i]   # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND 1d EMA34 rising AND volume confirmation
            if (breakout_up and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND 1d EMA34 falling AND volume confirmation
            elif (breakout_down and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR 1d EMA34 falls (trend change)
            if (curr_low < camarilla_s3[i] or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR 1d EMA34 rises (trend change)
            if (curr_high > camarilla_r3[i] or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals