#!/usr/bin/env python3
name = "6h_RVOL_Reversal_1dTrend"
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
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d RSI for trend confirmation
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h RVOL: volume / 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / np.where(vol_ma == 0, 1, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(rvol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RVOL > 2.0 + price below EMA50 + RSI < 40 (oversold in downtrend)
            if rvol[i] > 2.0 and close[i] < ema_50_1d_aligned[i] and rsi_1d_aligned[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: RVOL > 2.0 + price above EMA50 + RSI > 60 (overbought in uptrend)
            elif rvol[i] > 2.0 and close[i] > ema_50_1d_aligned[i] and rsi_1d_aligned[i] > 60:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RVOL > 2.0 + price crosses above EMA50 or RSI > 60
            if rvol[i] > 2.0 and (close[i] > ema_50_1d_aligned[i] or rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RVOL > 2.0 + price crosses below EMA50 or RSI < 40
            if rvol[i] > 2.0 and (close[i] < ema_50_1d_aligned[i] or rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals