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
    
    # Get daily data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 14-period RSI on daily (overbought/oversold)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-period Bollinger Bands on daily (volatility filter)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Align indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Calculate session filter (8 AM - 8 PM UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Mean reversion signals from Bollinger Bands
        near_lower_bb = close[i] < lower_bb_aligned[i] * 1.01  # Within 1% of lower band
        near_upper_bb = close[i] > upper_bb_aligned[i] * 0.99  # Within 1% of upper band
        
        # RSI conditions
        rsi_oversold = rsi_14_aligned[i] < 30
        rsi_overbought = rsi_14_aligned[i] > 70
        
        # Entry conditions
        long_entry = (near_lower_bb and rsi_oversold and above_ema)  # Oversold in uptrend
        short_entry = (near_upper_bb and rsi_overbought and below_ema)  # Overbought in downtrend
        
        # Exit conditions: opposite signal or trend reversal
        exit_long = position == 1 and (near_upper_bb or rsi_overbought or below_ema)
        exit_short = position == -1 and (near_lower_bb or rsi_oversold or above_ema)
        
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

name = "1h_1d_bb_rsi_ema_mean_reversion"
timeframe = "1h"
leverage = 1.0