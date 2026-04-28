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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ATR(10)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Weekly EMA(20) for trend
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly RSI(14) for momentum
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to daily timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    atr_10_aligned = align_htf_to_ltf(prices, df_1w, atr_10)
    
    # Daily Donchian channel breakout (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_10_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema_20_aligned[i]
        trend_down = close[i] < ema_20_aligned[i]
        
        # Momentum filter: RSI in favorable range
        rsi_momentum_up = rsi_aligned[i] > 50
        rsi_momentum_down = rsi_aligned[i] < 50
        
        # Breakout filters
        breakout_up = close[i] > high_20[i-1]  # Break above previous 20-day high
        breakout_down = close[i] < low_20[i-1]  # Break below previous 20-day low
        
        # Entry conditions
        long_entry = trend_up and rsi_momentum_up and breakout_up
        short_entry = trend_down and rsi_momentum_down and breakout_down
        
        # Exit conditions: opposite trend or RSI reversal
        long_exit = not trend_up or rsi_aligned[i] < 50 or close[i] < low_20[i]
        short_exit = not trend_down or rsi_aligned[i] > 50 or close[i] > high_20[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA20_RSI_Breakout"
timeframe = "1d"
leverage = 1.0