#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly RSI(14) for momentum filter
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.clip(lower=0)
    loss_1w = -delta_1w.clip(upper=0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_values = rsi_1w.fillna(50).values
    
    # Align weekly RSI to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Momentum filter: RSI in favorable range
        rsi_ok_long = rsi_1w_aligned[i] > 50
        rsi_ok_short = rsi_1w_aligned[i] < 50
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions: price crosses weekly EMA20 with momentum and volume
        long_entry = (close[i] > ema20_1w_aligned[i] and 
                     close[i-1] <= ema20_1w_aligned[i-1] and  # crossed above
                     rsi_ok_long and vol_filter)
        short_entry = (close[i] < ema20_1w_aligned[i] and 
                      close[i-1] >= ema20_1w_aligned[i-1] and  # crossed below
                      rsi_ok_short and vol_filter)
        
        # Exit conditions: opposite crossover or momentum reversal
        long_exit = (close[i] < ema20_1w_aligned[i] or 
                    rsi_1w_aligned[i] < 40)
        short_exit = (close[i] > ema20_1w_aligned[i] or 
                     rsi_1w_aligned[i] > 60)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyEMA20_RSI_Momentum"
timeframe = "12h"
leverage = 1.0