#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4h trend filter and volume confirmation.
# Long when: 1h RSI > 50, 4h EMA21 upward, volume > 1.5x 20-period average
# Short when: 1h RSI < 50, 4h EMA21 downward, volume > 1.5x 20-period average
# Exit when: 1h RSI crosses back through 50
# Uses 4h EMA21 for trend direction, 1h RSI for momentum entry, volume for confirmation.
# Target: 15-30 trades/year per symbol. Works in bull (buy momentum) and bear (sell weakness).
name = "1h_RSI50_EMA21_Trend_Volume"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA21 on 4h data for trend filter
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMA21 to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 1h RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for EMA21 and RSI calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema21 = ema21_4h_aligned[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: RSI > 50, EMA21 upward, volume spike
            if (rsi_val > 50 and rsi[i-1] <= 50 and 
                ema21 > ema21_4h_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI < 50, EMA21 downward, volume spike
            elif (rsi_val < 50 and rsi[i-1] >= 50 and 
                  ema21 < ema21_4h_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses back below 50
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses back above 50
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals