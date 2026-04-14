#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI Pullback with 4h Trend Filter and Volume Confirmation
# Takes long when: price pulls back to RSI(30) in a 4h uptrend with volume spike
# Takes short when: price bounces off RSI(70) in a 4h downtrend with volume spike
# Uses 4h EMA(50) for trend direction and 1h volume spike (>1.5x 20-period average)
# Designed to capture mean-reversion moves within established trends, avoiding counter-trend trades
# Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume average (20-period)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for RSI and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        rsi_val = rsi[i]
        trend = ema_4h_aligned[i]
        
        if position == 0:
            # Long setup: RSI < 30 (oversold) in uptrend with volume spike
            if (rsi_val < 30 and 
                price > trend and  # Uptrend filter
                vol_current > 1.5 * vol_ma[i]):  # Volume spike
                position = 1
                signals[i] = position_size
            # Short setup: RSI > 70 (overbought) in downtrend with volume spike
            elif (rsi_val > 70 and 
                  price < trend and  # Downtrend filter
                  vol_current > 1.5 * vol_ma[i]):  # Volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend breaks
            if rsi_val >= 50 or price < trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend breaks
            if rsi_val <= 50 or price > trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_Pullback_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0