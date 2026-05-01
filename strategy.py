#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w Supertrend(ATR=10,mult=3) trend filter and volume confirmation.
# Long when price breaks above upper Donchian band AND 1w Supertrend is bullish AND volume > 1.5x 20-bar average.
# Short when price breaks below lower Donchian band AND 1w Supertrend is bearish AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture medium-term trends.
# Donchian channels provide robust price channels that adapt to volatility.
# 1w Supertrend trend filter ensures alignment with higher timeframe momentum with built-in ATR-based stop.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Target: 30-100 total trades over 4 years (7-25/year) for BTC/ETH/SOL.

name = "1d_Donchian20_1wSupertrend_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w Supertrend calculation (ATR=10, mult=3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(10)
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2 + 3 * atr
    basic_lb = (high_1w + low_1w) / 2 - 3 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.copy(basic_ub)
    final_lb = np.copy(basic_lb)
    
    for i in range(1, len(close_1w)):
        if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.copy(final_ub)
    for i in range(1, len(close_1w)):
        if close_1w[i] <= final_ub[i]:
            supertrend[i] = final_ub[i]
        else:
            supertrend[i] = final_lb[i]
    
    # Supertrend direction: True = bullish (price > supertrend), False = bearish (price < supertrend)
    supertrend_dir = close_1w > supertrend
    
    # Align Supertrend direction to 1d timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir.astype(float))
    
    # Calculate Donchian channels (20-period) on 1d data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 1d volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 1d timeframe
        hour = hours[i]
        
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(supertrend_dir_aligned[i]) or np.isnan(vol_ma[i]):
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
            # Long: breakout above upper band AND 1w Supertrend bullish AND volume confirmation
            if (breakout_up and 
                supertrend_dir_aligned[i] > 0.5 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND 1w Supertrend bearish AND volume confirmation
            elif (breakout_down and 
                  supertrend_dir_aligned[i] <= 0.5 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR 1w Supertrend turns bearish
            if (curr_low < lowest_low[i] or 
                supertrend_dir_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR 1w Supertrend turns bullish
            if (curr_high > highest_high[i] or 
                supertrend_dir_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals