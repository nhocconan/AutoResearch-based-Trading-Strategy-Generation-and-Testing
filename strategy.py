#!/usr/bin/env python3
"""
Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
Long when price > 4h EMA50, RSI(14) > 55, and volume > 1.5x 20-bar average.
Short when price < 4h EMA50, RSI(14) < 45, and volume > 1.5x 20-bar average.
Exit when RSI returns to neutral zone (45-55) or opposite signal.
Uses 4h for trend direction to reduce whipsaw, 1h for precise entry/exit.
Target: 20-40 trades/year per symbol with controlled risk.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume confirmation: 20-bar average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50 and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma_20.iloc[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: price above 4h EMA50, bullish momentum, volume confirmation
            if price > ema_50_4h_aligned[i] and rsi_val > 55 and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA50, bearish momentum, volume confirmation
            elif price < ema_50_4h_aligned[i] and rsi_val < 45 and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or bearish reversal
            if rsi_val < 45:  # momentum broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral or bullish reversal
            if rsi_val > 55:  # momentum broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_RSI_Volume_Momentum"
timeframe = "1h"
leverage = 1.0