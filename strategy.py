#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI and 1d Trend Filter
# - Uses 4h RSI(14) for overbought/oversold conditions
# - Enters long when 4h RSI < 30 and price pulls back to 1h EMA(21) with volume confirmation
# - Enters short when 4h RSI > 70 and price pulls back to 1h EMA(21) with volume confirmation
# - Uses 1d EMA(50) to filter trades in direction of daily trend
# - Uses volume spike (volume > 1.5x 20-period average) for entry confirmation
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_RSI40_EMA21_TrendFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(df_4h['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 4h RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h EMA(21) for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after warmup for EMA21
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 4h RSI oversold (<30) + price at 1h EMA21 + volume + above daily EMA50
            if (rsi_4h_aligned[i] < 30 and 
                close[i] <= ema_21[i] * 1.002 and  # Allow small tolerance above EMA
                close[i] >= ema_21[i] * 0.998 and   # Allow small tolerance below EMA
                volume_filter[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: 4h RSI overbought (>70) + price at 1h EMA21 + volume + below daily EMA50
            elif (rsi_4h_aligned[i] > 70 and 
                  close[i] >= ema_21[i] * 0.998 and  # Allow small tolerance below EMA
                  close[i] <= ema_21[i] * 1.002 and  # Allow small tolerance above EMA
                  volume_filter[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA21 or RSI reaches overbought
            if close[i] < ema_21[i] * 0.995 or rsi_4h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above EMA21 or RSI reaches oversold
            if close[i] > ema_21[i] * 1.005 or rsi_4h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals