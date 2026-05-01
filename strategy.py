#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
# Long when price breaks above upper Donchian channel AND 1d EMA50 rising AND ATR(14) < ATR(50) (low volatility environment).
# Short when price breaks below lower Donchian channel AND 1d EMA50 falling AND ATR(14) < ATR(50).
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to capture medium-term trends with volatility filter to avoid false breakouts in choppy markets.
# Donchian channels provide clear breakout levels that work in both trending and ranging markets when combined with volatility filter.
# 1d EMA50 trend filter ensures alignment with higher timeframe momentum to avoid counter-trend trades.
# ATR ratio filter ensures we only trade in low volatility environments where breakouts are more likely to succeed.

name = "4h_Donchian20_1dEMA50_ATRVolFilter_v1"
timeframe = "4h"
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
    
    # Donchian(20) channels on 4h data
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR-based volatility filter: ATR(14) < ATR(50) indicates low volatility environment
    atr_period_short = 14
    atr_period_long = 50
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = high[0] - close[0]  # First bar
    tr3[0] = high[0] - low[0]  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=atr_period_short, min_periods=atr_period_short).mean().values
    atr_50 = pd.Series(tr).rolling(window=atr_period_long, min_periods=atr_period_long).mean().values
    atr_ratio = atr_14 / atr_50
    low_volatility = atr_ratio < 1.0  # ATR(14) < ATR(50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, atr_period_long) + 5  # warmup
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Donchian breakout signals
        breakout_up = curr_high > upper_channel[i]  # break above upper channel
        breakout_down = curr_low < lower_channel[i]  # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper channel AND 1d EMA50 rising AND low volatility
            if (breakout_up and 
                ema_50_rising[i] and 
                low_volatility[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel AND 1d EMA50 falling AND low volatility
            elif (breakout_down and 
                  ema_50_falling[i] and 
                  low_volatility[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower channel (stoploss) OR 1d EMA50 falls (trend change)
            if (curr_low < lower_channel[i] or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel (stoploss) OR 1d EMA50 rises (trend change)
            if (curr_high > upper_channel[i] or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals