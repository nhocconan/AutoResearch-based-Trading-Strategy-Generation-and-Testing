#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_ema_rsi_weekly_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Calculate daily EMA(20) for momentum
    ema_fast = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: daily volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_fast[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA(50)
        trend_up = close[i] > weekly_ema_aligned[i]
        trend_down = close[i] < weekly_ema_aligned[i]
        
        # Momentum: EMA(20) direction
        ema_up = ema_fast[i] > ema_fast[i-1]
        ema_down = ema_fast[i] < ema_fast[i-1]
        
        # RSI conditions
        rsi_overbought = rsi_values[i] > 70
        rsi_oversold = rsi_values[i] < 30
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit on trend reversal or overbought RSI
            if not trend_up or rsi_overbought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on trend reversal or oversold RSI
            if not trend_down or rsi_oversold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: uptrend + EMA rising + oversold RSI + volume
            if trend_up and ema_up and rsi_oversold and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: downtrend + EMA falling + overbought RSI + volume
            elif trend_down and ema_down and rsi_overbought and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals