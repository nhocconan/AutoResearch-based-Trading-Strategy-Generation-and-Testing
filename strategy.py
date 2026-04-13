#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 10-period RSI on 1d (momentum filter)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-period ATR on 1d (volatility filter)
    tr1 = np.zeros(len(high_1d))
    tr2 = np.zeros(len(high_1d))
    tr3 = np.zeros(len(high_1d))
    tr1[1:] = high_1d[1:] - low_1d[1:]
    tr2[1:] = np.abs(high_1d[1:] - close_1d[:-1])
    tr3[1:] = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1d = np.zeros_like(tr)
    for i in range(20, len(tr)):
        atr_20_1d[i] = np.mean(tr[i-20:i])
    
    # Calculate 50-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period Donchian channels on 6h
    high_20 = np.full(len(high), np.nan)
    low_20 = np.full(len(low), np.nan)
    for i in range(20, len(high)):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    # Align indicators to 6h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_20_1d_aligned[i]) or
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_14_1d_aligned[i] < 70
        rsi_not_oversold = rsi_14_1d_aligned[i] > 30
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_20_1d_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20[i]
        short_breakout = close[i] < low_20[i]
        
        # Entry conditions: breakout with trend and momentum alignment
        long_entry = long_breakout and above_ema and rsi_not_overbought and vol_filter
        short_entry = short_breakout and below_ema and rsi_not_oversold and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_ema)
        exit_short = position == -1 and (long_breakout or above_ema)
        
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

name = "6h_1d_rsi_ema_donchian_breakout"
timeframe = "6h"
leverage = 1.0