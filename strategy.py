#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum with 4h trend filter and 1d volume regime filter
    # Long: 1h RSI > 55 + 4h close > 4h EMA20 + 1d volume > 1.5x 20-day average
    # Short: 1h RSI < 45 + 4h close < 4h EMA20 + 1d volume > 1.5x 20-day average
    # Uses discrete sizing (0.20) to control risk and minimize fee churn
    # Target: 15-37 trades/year to stay within 1h optimal range (60-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d 20-period volume average for regime filter
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: 1d volume > 1.5x 20-day average (high conviction days)
        high_volume_regime = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 4h close above/below EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Momentum conditions: RSI extremes with volume and trend alignment
        long_signal = (rsi[i] > 55) and high_volume_regime and uptrend
        short_signal = (rsi[i] < 45) and high_volume_regime and downtrend
        
        # Execute signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and not (rsi[i] > 50 and uptrend):
            # Exit long if momentum fades
            position = 0
            signals[i] = 0.0
        elif position == -1 and not (rsi[i] < 50 and downtrend):
            # Exit short if momentum fades
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

name = "1h_4h_1d_rsi_momentum_volume_regime_v1"
timeframe = "1h"
leverage = 1.0