#!/usr/bin/env python3
name = "1D_1W_RSI20_Pullback_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(20) for trend filter
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.clip(lower=0)
    loss_1w = -delta_1w.clip(upper=0)
    avg_gain_1w = gain_1w.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    rs_1w = avg_gain_1w / avg_loss_1w.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_values = rsi_1w.fillna(50).values
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Get daily data for RSI(14) and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14)
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.fillna(50).values
    
    # Calculate daily volume SMA(10)
    vol_sma10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_1d_values[i]) or np.isnan(vol_sma10_1d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Weekly trend: RSI(20) > 50 = bullish, < 50 = bearish
        weekly_bullish = rsi_1w_aligned[i] > 50
        weekly_bearish = rsi_1w_aligned[i] < 50
        # Daily oversold/overbought: RSI(14) < 30 = oversold, > 70 = overbought
        daily_oversold = rsi_1d_values[i] < 30
        daily_overbought = rsi_1d_values[i] > 70
        # Volume confirmation: current volume > 1.5x 10-day average
        volume_confirm = volume[i] > vol_sma10_1d[i] * 1.5
        
        if position == 0:
            # Enter long: Weekly bullish + daily oversold + volume confirmation
            if weekly_bullish and daily_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly bearish + daily overbought + volume confirmation
            elif weekly_bearish and daily_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly turns bearish OR daily RSI > 50
            if not weekly_bullish or rsi_1d_values[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly turns bullish OR daily RSI < 50
            if not weekly_bearish or rsi_1d_values[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals