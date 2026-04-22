# 2024-06-25 13:30:00
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# RSI < 30 for long, RSI > 70 for short with 4h EMA(50) trend filter to avoid counter-trend trades
# Volume > 1.3x average confirms momentum. Designed for range-bound markets with clear trends.
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA trend (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + above 4h EMA + volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.3 * vol_avg_24[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + below 4h EMA + volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.3 * vol_avg_24[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone or trend reverses
            if position == 1:
                # Exit long: RSI > 50 or price below 4h EMA
                if (rsi[i] > 50 or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: RSI < 50 or price above 4h EMA
                if (rsi[i] < 50 or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0