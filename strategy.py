#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation.
# Long when price breaks above upper Donchian band AND 1w HMA21 rising AND volume > 1.5x 20-bar average.
# Short when price breaks below lower Donchian band AND 1w HMA21 falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture medium-term trends.
# HMA (Hull Moving Average) provides smoother trend with less lag than EMA/SMA.
# Donchian channels provide robust price channels that adapt to volatility.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Target: 30-100 total trades over 4 years (7-25/year) for BTC/ETH/SOL.

name = "1d_Donchian20_1wHMA21_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # 1w HMA21 calculation
    close_1w = df_1w['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    raw_2wma = 2 * wma_half
    raw_2wma_padded = np.full(len(close_1w), np.nan)
    raw_2wma_padded[half_len-1:half_len-1+len(raw_2wma)] = raw_2wma
    diff = raw_2wma_padded - wma_full
    hma_21 = wma(diff, sqrt_len)
    hma_21_padded = np.full(len(close_1w), np.nan)
    hma_21_padded[sqrt_len-1:sqrt_len-1+len(hma_21)] = hma_21
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    
    # 1w HMA21 slope (rising/falling)
    hma_21_slope = np.diff(hma_21_aligned, prepend=hma_21_aligned[0])
    hma_21_rising = hma_21_slope > 0
    hma_21_falling = hma_21_slope < 0
    
    # Calculate Donchian channels (20-period) on 1d data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 1d volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for HMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 1d timeframe
        hour = hours[i]
        
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > highest_high[i]  # break above upper band
        breakout_down = curr_low < lowest_low[i]   # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND 1w HMA21 rising AND volume confirmation
            if (breakout_up and 
                hma_21_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND 1w HMA21 falling AND volume confirmation
            elif (breakout_down and 
                  hma_21_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR 1w HMA21 falls (trend change)
            if (curr_low < lowest_low[i] or 
                hma_21_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR 1w HMA21 rises (trend change)
            if (curr_high > highest_high[i] or 
                hma_21_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals