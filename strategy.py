#!/usr/bin/env python3
"""
1h_4h_Trend_1h_Momentum_v1
1h strategy using 4h EMA trend filter and 1h momentum confirmation.
- Trend: 4h EMA50 > EMA200 for long, EMA50 < EMA200 for short
- Entry: 1h RSI(14) crossing above 55 (long) or below 45 (short) with volume confirmation
- Exit: Opposite RSI cross or trend reversal
- Volume: 1h volume > 1.5x 20-period average
- Session filter: 08-20 UTC only
- Position size: 0.20
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # === 4h EMA Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMAs on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Trend direction: 1 = uptrend, -1 = downtrend, 0 = no trend
    trend_4h = np.where(ema50_4h_aligned > ema200_4h_aligned, 1,
                        np.where(ema50_4h_aligned < ema200_4h_aligned, -1, 0))
    
    # === 1h RSI for Momentum Entry ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1h Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry signals
        rsi_long_signal = (rsi[i] > 55) and (rsi[i-1] <= 55)  # Cross above 55
        rsi_short_signal = (rsi[i] < 45) and (rsi[i-1] >= 45)  # Cross below 45
        
        # Trend alignment
        uptrend = trend_4h[i] == 1
        downtrend = trend_4h[i] == -1
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: uptrend + RSI crosses above 55 + volume
            if uptrend and rsi_long_signal and vol_confirmed:
                signals[i] = 0.20
                position = 1
                continue
            # Short: downtrend + RSI crosses below 45 + volume
            elif downtrend and rsi_short_signal and vol_confirmed:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses below 45 OR trend turns down
            if (rsi[i] < 45 and rsi[i-1] >= 45) or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses above 55 OR trend turns up
            if (rsi[i] > 55 and rsi[i-1] <= 55) or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_Trend_1h_Momentum_v1"
timeframe = "1h"
leverage = 1.0