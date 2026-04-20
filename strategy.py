#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Uses 4h RSI(14) for trend direction (RSI > 55 = uptrend, RSI < 45 = downtrend) 
# and 1h RSI(14) for entry timing (RSI < 30 for long, RSI > 70 for short).
# Volume confirmation requires current volume > 1.5x 20-period average.
# Session filter: only trade 08:00-20:00 UTC to avoid low-liquidity hours.
# This combines mean reversion in ranging markets with trend filtering to avoid 
# counter-trend trades during strong moves. Designed for 15-30 trades/year.
# Target: 60-120 total trades over 4 years.

name = "1h_RSI_MeanReversion_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === 4h RSI(14) for trend direction ===
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.fillna(50).values
    rsi_4h_trend = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # === 1h RSI(14) for entry timing ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    # === Session filter: 08:00-20:00 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after RSI warmup
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        rsi_val = rsi[i]
        rsi_4h_val = rsi_4h_trend[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(rsi_4h_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend (RSI > 55) + 1h oversold (RSI < 30) + volume
            if rsi_4h_val > 55 and rsi_val < 30 and vol_ratio_val > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (RSI < 45) + 1h overbought (RSI > 70) + volume
            elif rsi_4h_val < 45 and rsi_val > 70 and vol_ratio_val > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 1h RSI > 50 (mean reversion complete) or trend change
            if rsi_val > 50 or rsi_4h_val < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 1h RSI < 50 (mean reversion complete) or trend change
            if rsi_val < 50 or rsi_4h_val > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals