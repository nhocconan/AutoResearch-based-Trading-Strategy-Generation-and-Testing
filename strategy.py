# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly RSI as filter with daily RSI mean reversion
# - Long when weekly RSI > 50 (bullish regime) and daily RSI < 30 (oversold)
# - Short when weekly RSI < 50 (bearish regime) and daily RSI > 70 (overbought)
# - Weekly RSI determines market regime, daily RSI provides mean-reversion entries
# - Volume confirmation ensures institutional participation at reversal points
# - Target: 30-80 total trades over 4 years (7-20/year) to balance opportunity and cost
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period)
    close_w = df_w['close'].values
    delta_w = np.diff(close_w, prepend=close_w[0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    avg_gain_w = pd.Series(gain_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_w = pd.Series(loss_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_w = avg_gain_w / (avg_loss_w + 1e-10)
    rsi_w = 100 - (100 / (1 + rs_w))
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get weekly index for current daily bar
        idx_w = i // 7  # Approximate weekly index from daily bars
        if idx_w < 1:
            continue
            
        # Previous week's RSI (to avoid look-ahead)
        rsi_w_prev = rsi_w[idx_w-1]
        
        # Create arrays for alignment (constant values for the week)
        rsi_w_arr = np.full(len(df_w), rsi_w_prev)
        
        # Align to daily timeframe
        rsi_w_daily = align_htf_to_ltf(prices, df_w, rsi_w_arr)[i]
        
        if position == 0:
            # Long: Weekly RSI > 50 (bullish regime) + daily RSI < 30 (oversold) + volume
            if (rsi_w_daily > 50 and  # Weekly bullish regime
                rsi[i] < 30 and       # Daily oversold
                volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: Weekly RSI < 50 (bearish regime) + daily RSI > 70 (overbought) + volume
            elif (rsi_w_daily < 50 and  # Weekly bearish regime
                  rsi[i] > 70 and       # Daily overbought
                  volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Daily RSI > 50 (mean reversion complete) or weekly RSI < 40 (regime change)
            if rsi[i] > 50 or rsi_w_daily < 40:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Daily RSI < 50 (mean reversion complete) or weekly RSI > 60 (regime change)
            if rsi[i] < 50 or rsi_w_daily > 60:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0