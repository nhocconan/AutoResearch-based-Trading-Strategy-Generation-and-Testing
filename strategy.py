#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 14-period RSI with weekly trend filter and volume confirmation
# We go long when RSI crosses above 30 (oversold bounce) with weekly EMA(34) uptrend and volume spike.
# We go short when RSI crosses below 70 (overbought rejection) with weekly EMA(34) downtrend and volume spike.
# Designed for low trade frequency in both bull and bear markets with proper risk control.
# Target: 30-100 total trades over 4 years = 7-25/year (within 150 max)

name = "1d_RSI14_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 14-period RSI on daily data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike: current volume > 2.0 * 14-period average
    vol_ma = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        rsi_val = rsi_values[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: RSI crosses above 30 + weekly uptrend + volume spike
            if (rsi_val > 30 and rsi_values[i-1] <= 30 and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI crosses below 70 + weekly downtrend + volume spike
            elif (rsi_val < 70 and rsi_values[i-1] >= 70 and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses below 50 OR weekly trend turns down
            if (rsi_val < 50 and rsi_values[i-1] >= 50) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses above 50 OR weekly trend turns up
            if (rsi_val > 50 and rsi_values[i-1] <= 50) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals