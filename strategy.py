#!/usr/bin/env python3
"""
1d_200EMA_RSI_Trend_Volume
Hypothesis: Uses 200-day EMA for long-term trend direction, RSI(14) for momentum confirmation, and volume spikes for entry timing.
Trades long when price is above 200EMA with RSI > 50 and volume spike, short when below 200EMA with RSI < 50 and volume spike.
Designed to capture major trends while avoiding choppy markets. Targets 7-25 trades per year to minimize fee drag.
"""

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
    volume = prices['volume'].values
    
    # Get 1-day data for 200EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate RSI(14) on 1-day data
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Calculate volume spike (>1.5x 20-period MA for entry timing)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA200
        trend_up = close[i] > ema_200_1d_aligned[i]
        trend_down = close[i] < ema_200_1d_aligned[i]
        
        # Momentum from RSI
        rsi_bullish = rsi_1d_aligned[i] > 50
        rsi_bearish = rsi_1d_aligned[i] < 50
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic:
        # Long: Above 200EMA, RSI > 50, volume spike
        long_entry = vol_confirm and trend_up and rsi_bullish
        # Short: Below 200EMA, RSI < 50, volume spike
        short_entry = vol_confirm and trend_down and rsi_bearish
        
        # Exit logic: Opposite trend or RSI extreme
        long_exit = not trend_up or rsi_1d_aligned[i] < 30  # Exit on trend change or oversold
        short_exit = not trend_down or rsi_1d_aligned[i] > 70  # Exit on trend change or overbought
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_200EMA_RSI_Trend_Volume"
timeframe = "1d"
leverage = 1.0