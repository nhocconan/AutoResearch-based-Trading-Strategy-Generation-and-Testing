#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_RSI14_Trend_WeeklyBias"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    Daily RSI(14) with weekly trend bias and volume confirmation.
    - Long: RSI < 30 (oversold) + price above weekly EMA50 + volume spike
    - Short: RSI > 70 (overbought) + price below weekly EMA50 + volume spike
    - Exit when RSI returns to neutral zone (40-60)
    - Target: 15-25 trades/year on 1d timeframe
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14) on daily closes
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike detection (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema50_1d[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: RSI oversold (<30) + price above weekly EMA50 + volume spike
            if (rsi_values[i] < 30 and close[i] > ema50_1d[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price below weekly EMA50 + volume spike
            elif (rsi_values[i] > 70 and close[i] < ema50_1d[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>=40) or weekly trend turns down
            if rsi_values[i] >= 40 or close[i] < ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<=60) or weekly trend turns up
            if rsi_values[i] <= 60 or close[i] > ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals