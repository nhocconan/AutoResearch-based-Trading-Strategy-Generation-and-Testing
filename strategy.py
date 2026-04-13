#!/usr/bin/env python3
"""
Hypothesis: 12h EMA trend filter with 1d RSI mean reversion and volume confirmation.
Uses 12h EMA(50) for trend direction, 1d RSI(14) for overbought/oversold conditions, 
and 1d volume spike (volume > 1.5x 20-period average) to confirm momentum.
Long when price above 12h EMA(50), 1d RSI < 30, and volume spike. 
Short when price below 12h EMA(50), 1d RSI > 70, and volume spike.
Exit when RSI returns to neutral zone (40-60).
Designed for low-frequency, high-conviction trades to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h EMA(50) for trend filter
    ema_12h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for RSI and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    
    # Align 1d indicators to 12h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_12h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        long_entry = price_above_ema and rsi_oversold and vol_confirm
        short_entry = price_below_ema and rsi_overbought and vol_confirm
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and (rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60)
        exit_short = position == -1 and (rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_ema_rsi_vol_meanrev"
timeframe = "12h"
leverage = 1.0