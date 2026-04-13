#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period SMA on 1w
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Calculate ATR(14) on 4h for volatility filter and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20) on 4h for breakout signals
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA20 (1d)
        above_ema = close[i] > ema_20_aligned[i]
        below_ema = close[i] < ema_20_aligned[i]
        
        # RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Weekly trend filter: price above/below weekly SMA20
        above_weekly_sma = close[i] > sma_20_1w_aligned[i]
        below_weekly_sma = close[i] < sma_20_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above 20-period high
        breakout_down = close[i] < lowest_low[i-1]   # Break below 20-period low
        
        # Volatility filter: require minimum volatility to avoid choppy markets
        vol_filter = atr[i] > 0.01 * close[i]  # At least 1% of price as ATR
        
        # Entry conditions
        long_entry = (above_ema and rsi_not_overbought and above_weekly_sma and 
                     breakout_up and vol_filter)
        short_entry = (below_ema and rsi_not_oversold and below_weekly_sma and 
                      breakout_down and vol_filter)
        
        # Exit conditions: opposite signal or volatility drop
        exit_long = position == 1 and (below_ema or not vol_filter)
        exit_short = position == -1 and (above_ema or not vol_filter)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_ema_rsi_breakout"
timeframe = "4h"
leverage = 1.0