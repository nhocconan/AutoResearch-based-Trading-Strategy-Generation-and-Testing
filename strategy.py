#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day linear regression slope as trend filter and 12-hour RSI for mean-reversion entries.
# - Long when price is above 1-day linear regression trend (bullish) and 12h RSI < 30 (oversold)
# - Short when price is below 1-day linear regression trend (bearish) and 12h RSI > 70 (overbought)
# - Volume confirmation: current volume > 1.5x 20-period average to ensure participation
# - Uses 1-day linear regression for robust trend filtering that adapts to price direction
# - RSI(14) provides mean-reversion signals within the trend context
# - Target: 50-150 total trades over 4 years (12-37/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day linear regression slope and intercept
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    x = np.arange(n_1d)
    # Calculate slope and intercept using least squares
    sum_x = np.sum(x)
    sum_y = np.sum(close_1d)
    sum_xy = np.sum(x * close_1d)
    sum_x2 = np.sum(x * x)
    
    slope = np.full(n_1d, np.nan)
    intercept = np.full(n_1d, np.nan)
    
    # Calculate for each point with sufficient history
    for i in range(19, n_1d):  # Need at least 20 points for regression
        # Use last 20 points for regression
        start_idx = i - 19
        x_subset = x[start_idx:i+1]
        y_subset = close_1d[start_idx:i+1]
        n_sub = len(x_subset)
        
        sum_x_sub = np.sum(x_subset)
        sum_y_sub = np.sum(y_subset)
        sum_xy_sub = np.sum(x_subset * y_subset)
        sum_x2_sub = np.sum(x_subset * x_subset)
        
        denominator = n_sub * sum_x2_sub - sum_x_sub * sum_x_sub
        if denominator != 0:
            slope[i] = (n_sub * sum_xy_sub - sum_x_sub * sum_y_sub) / denominator
            intercept[i] = (sum_y_sub - slope[i] * sum_x_sub) / n_sub
    
    # Calculate linear regression values (trend line)
    lr_values = slope * x + intercept
    
    # Calculate 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 12h bar (12h = 0.5 * 1d)
        idx_1d = i // 2
        if idx_1d < 20:  # Need sufficient 1d data for regression
            continue
            
        # Previous 1d linear regression value (to avoid look-ahead)
        lr_prev = lr_values[idx_1d-1]
        if np.isnan(lr_prev):
            continue
        
        # Create arrays for alignment (constant values for the 1d period)
        lr_arr = np.full(len(df_1d), lr_prev)
        
        # Align to 12h timeframe
        lr_12h = align_htf_to_ltf(prices, df_1d, lr_arr)[i]
        
        if position == 0:
            # Long: price above 1d LR trend + RSI oversold + volume confirmation
            if (close[i] > lr_12h and  # price above 1d linear regression
                rsi[i] < 30 and  # RSI oversold
                volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price below 1d LR trend + RSI overbought + volume confirmation
            elif (close[i] < lr_12h and  # price below 1d linear regression
                  rsi[i] > 70 and  # RSI overbought
                  volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: RSI overbought or price below 1d LR trend
            if rsi[i] > 70 or close[i] < lr_12h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: RSI oversold or price above 1d LR trend
            if rsi[i] < 30 or close[i] > lr_12h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_LinearRegression_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0