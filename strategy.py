#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI mean reversion with 4-hour trend filter and volume confirmation
# Long when RSI(14) < 30, price > 4h EMA50 (uptrend), and volume > 1.5x 20-period average
# Short when RSI(14) > 70, price < 4h EMA50 (downtrend), and volume > 1.5x 20-period average
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit)
# Uses RSI for mean reversion extremes, 4h EMA for trend alignment, volume for confirmation
# Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC) to reduce noise

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations (14 for RSI + buffer)
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: RSI < 30 (oversold), price > 4h EMA50 (uptrend), volume confirmation
            if (rsi_val < 30 and price > ema50_4h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI > 70 (overbought), price < 4h EMA50 (downtrend), volume confirmation
            elif (rsi_val > 70 and price < ema50_4h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back to 40 (mean reversion complete)
            if rsi_val >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses back to 60 (mean reversion complete)
            if rsi_val <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0