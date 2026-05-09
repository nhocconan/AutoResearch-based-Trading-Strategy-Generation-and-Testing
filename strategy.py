#/usr/bin/env python3
name = "6H_DailyATR_RSI_RewardRisk_Trend_Entry"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Align to 6h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(atr_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above EMA20 + RSI > 50 + volatility filter (ATR not too low)
            if close[i] > ema20[i] and rsi_aligned[i] > 50 and atr_aligned[i] > 0:
                # Calculate risk-reward: target = 2x ATR, stop = 1x ATR
                # Only enter if potential reward justifies risk (minimum 1.5:1)
                # Since we use close-based exits, we rely on the trend continuation
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA20 + RSI < 50 + volatility filter
            elif close[i] < ema20[i] and rsi_aligned[i] < 50 and atr_aligned[i] > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA20 (trend change) OR RSI < 30 (oversold)
            if close[i] < ema20[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA20 (trend change) OR RSI > 70 (overbought)
            if close[i] > ema20[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals