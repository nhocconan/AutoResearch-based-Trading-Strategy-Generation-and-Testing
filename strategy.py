#!/usr/bin/env python3
"""
12h_1d_RSI_Extreme_Reversal
Hypothesis: In a bear market (2025-2026), extreme RSI readings on the daily timeframe signal exhaustion and imminent reversals. 
On 12h timeframe, look for RSI > 85 (overbought) or RSI < 15 (oversold) on the daily chart as reversal signals. 
Enter short on daily RSI > 85 with price below 20-period EMA on 12h; enter long on daily RSI < 15 with price above 20-period EMA on 12h. 
Use volume spike (2.0x 20-period average) to confirm institutional participation. 
Exit on RSI returning to neutral zone (40-60) or opposite extreme. 
Designed to capture mean-reversion moves in bear markets with low trade frequency to avoid fee drag.
"""

name = "12h_1d_RSI_Extreme_Reversal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to RMA)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.finfo(float).eps, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 12h EMA(20) for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: daily RSI < 15 (oversold) + price above 12h EMA20 + volume spike
            if (rsi_aligned[i] < 15 and 
                close[i] > ema_20[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: daily RSI > 85 (overbought) + price below 12h EMA20 + volume spike
            elif (rsi_aligned[i] > 85 and 
                  close[i] < ema_20[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone (40-60) or opposite extreme
            if position == 1:
                # Exit long: RSI returns to >= 40 or reaches overbought (>85)
                if (rsi_aligned[i] >= 40) or (rsi_aligned[i] > 85):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to <= 60 or reaches oversold (<15)
                if (rsi_aligned[i] <= 60) or (rsi_aligned[i] < 15):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals