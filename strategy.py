#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour Exponential Moving Average (EMA50) as trend filter
# with 4-hour Relative Strength Index (RSI) for mean-reversion entries.
# - Long when price is above 12h EMA50 (bullish trend) and 4h RSI < 30 (oversold)
# - Short when price is below 12h EMA50 (bearish trend) and 4h RSI > 70 (overbought)
# - Volume confirmation: current volume > 1.5x 20-period average to ensure participation
# - Uses 12h EMA50 for robust trend filtering that adapts to longer-term trends
# - RSI(14) provides mean-reversion signals within the trend context
# - Target: 75-200 total trades over 4 years (19-50/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h RSI(14)
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
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_12h).any():
            continue
        
        # Get 12h index for current 4h bar
        # 12h = 3 * 4h bars
        idx_12h = i // 3
        if idx_12h < 1:
            continue
            
        # Previous 12h EMA50 (to avoid look-ahead)
        ema_prev = ema_12h[idx_12h-1]
        
        # Create arrays for alignment (constant values for the 12h period)
        ema_arr = np.full(len(df_12h), ema_prev)
        
        # Align to 4h timeframe
        ema_4h = align_htf_to_ltf(prices, df_12h, ema_arr)[i]
        
        if position == 0:
            # Long: price above 12h EMA50 + RSI oversold + volume confirmation
            if (close[i] > ema_4h and  # price above 12h EMA50
                rsi[i] < 30 and  # RSI oversold
                volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price below 12h EMA50 + RSI overbought + volume confirmation
            elif (close[i] < ema_4h and  # price below 12h EMA50
                  rsi[i] > 70 and  # RSI overbought
                  volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: RSI overbought or price below 12h EMA50
            if rsi[i] > 70 or close[i] < ema_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: RSI oversold or price above 12h EMA50
            if rsi[i] < 30 or close[i] > ema_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA50_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0