#!/usr/bin/env python3
# Hypothesis: 1h momentum strategy with 4h trend filter and 1d volume confirmation
# Long when price breaks above 4h EMA50 with 1d volume > 1.5x 20-day average and RSI(14) > 55
# Short when price breaks below 4h EMA50 with 1d volume > 1.5x 20-day average and RSI(14) < 45
# Exit when price crosses back over 4h EMA50 or RSI reaches extreme levels (70/30)
# Uses 4h EMA for trend direction, 1d volume for conviction, 1h RSI for entry timing
# Designed to capture momentum moves in both trending and ranging markets with controlled frequency
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20

name = "1h_EMA50_RSI_VolumeFilter_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA/RSI calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume confirmation: current 1d volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma_1d_aligned[i])
            
            # Enter long: price above 4h EMA50, RSI > 55, volume confirmation
            if (close[i] > ema50_4h_aligned[i] and 
                rsi_values[i] > 55 and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA50, RSI < 45, volume confirmation
            elif (close[i] < ema50_4h_aligned[i] and 
                  rsi_values[i] < 45 and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 or RSI > 70 (overbought)
            if (close[i] < ema50_4h_aligned[i]) or (rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 or RSI < 30 (oversold)
            if (close[i] > ema50_4h_aligned[i]) or (rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals