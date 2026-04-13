#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for higher timeframe trend and momentum
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA200 for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly RSI for momentum
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Calculate daily volume average for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Align daily volume MA to daily timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(200, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long-term trend filter: price above/below weekly EMA200
        above_long_trend = close[i] > ema200_1w_aligned[i]
        below_long_trend = close[i] < ema200_1w_aligned[i]
        
        # Weekly momentum filter: RSI not extreme
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Volume filter: current volume above 20-day average
        volume_filter = volume[i] > vol_ma_20_aligned[i]
        
        # Entry conditions
        long_entry = above_long_trend and rsi_not_overbought and volume_filter
        short_entry = below_long_trend and rsi_not_oversold and volume_filter
        
        # Exit conditions: trend reversal or momentum extreme
        exit_long = position == 1 and (not above_long_trend or rsi_1w_aligned[i] >= 70)
        exit_short = position == -1 and (not below_long_trend or rsi_1w_aligned[i] <= 30)
        
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

name = "1d_1w_ema200_rsi_volume_filter"
timeframe = "1d"
leverage = 1.0