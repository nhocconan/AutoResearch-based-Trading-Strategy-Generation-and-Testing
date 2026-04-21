#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 50-period SMA trend with 20-period RSI pullback entries and volume confirmation.
# In uptrend (price > SMA50), look for RSI < 30 pullbacks with volume > 1.5x average to go long.
# In downtrend (price < SMA50), look for RSI > 70 pullbacks with volume > 1.5x average to go short.
# Uses weekly trend filter to ensure alignment with higher timeframe momentum.
# Target: 15-30 trades/year by requiring trend alignment + RSI extreme + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    close_w = df_1w['close'].values
    sma50_w = pd.Series(close_w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_w)
    
    # Load daily data for SMA50 and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily SMA50
    close_d = df_1d['close'].values
    sma50_d = pd.Series(close_d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_d)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close_d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(sma50_1w_aligned[i]) or np.isnan(sma50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        # Trend filters
        weekly_uptrend = price > sma50_1w_aligned[i]
        weekly_downtrend = price < sma50_1w_aligned[i]
        daily_uptrend = price > sma50_1d_aligned[i]
        daily_downtrend = price < sma50_1d_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        if position == 0:
            # Enter long in weekly uptrend with daily uptrend, RSI oversold and volume confirmation
            if weekly_uptrend and daily_uptrend and rsi_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short in weekly downtrend with daily downtrend, RSI overbought and volume confirmation
            elif weekly_downtrend and daily_downtrend and rsi_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral territory (40-60)
            exit_signal = False
            
            if position == 1:
                # Exit long when RSI >= 40
                if rsi_1d_aligned[i] >= 40:
                    exit_signal = True
            elif position == -1:
                # Exit short when RSI <= 60
                if rsi_1d_aligned[i] <= 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_SMA50_RSI_Pullback_Volume"
timeframe = "1d"
leverage = 1.0