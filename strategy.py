#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w KAMA trend filter and volume confirmation.
# Long when price breaks above upper Bollinger band AND 1w KAMA rising AND volume > 1.5x 20-bar average.
# Short when price breaks below lower Bollinger band AND 1w KAMA falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture medium-term trends.
# Bollinger Bands adapt to volatility and provide dynamic support/resistance.
# 1w KAMA trend filter ensures alignment with higher timeframe momentum with less lag.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Target: 30-100 total trades over 4 years (7-25/year) for BTC/ETH/SOL.

name = "1d_BB20_2_1wKAMA_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w KAMA calculation (adaptive moving average)
    close_1w = df_1w['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # 1w KAMA slope (rising/falling)
    kama_slope = np.diff(kama_aligned, prepend=kama_aligned[0])
    kama_rising = kama_slope > 0
    kama_falling = kama_slope < 0
    
    # Calculate Bollinger Bands (20, 2) on 1d data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Volume confirmation: current 1d volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 1d timeframe
        hour = hours[i]
        
        if np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Bollinger Band breakout signals
        breakout_up = curr_high > upper_bb[i]  # break above upper band
        breakout_down = curr_low < lower_bb[i]   # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND 1w KAMA rising AND volume confirmation
            if (breakout_up and 
                kama_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND 1w KAMA falling AND volume confirmation
            elif (breakout_down and 
                  kama_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR 1w KAMA falls (trend change)
            if (curr_low < lower_bb[i] or 
                kama_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR 1w KAMA rises (trend change)
            if (curr_high > upper_bb[i] or 
                kama_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals