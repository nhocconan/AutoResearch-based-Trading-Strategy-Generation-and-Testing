#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) pullback to 4h EMA(50) with 1d trend filter and volume confirmation
# RSI < 30 in uptrend or > 70 in downtrend identifies overextended moves ready for mean reversion.
# 4h EMA50 provides dynamic support/resistance in trending markets.
# 1d EMA100 filter ensures trades align with higher timeframe trend.
# Volume > 1.5x 20-period average confirms momentum behind the move.
# Designed for 1h timeframe targeting 15-30 trades/year with controlled risk in bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) - well-established momentum oscillator
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for EMA(50) dynamic support/resistance (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for EMA(100) trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above 4h EMA50 + 1d uptrend + volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_100_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + price below 4h EMA50 + 1d downtrend + volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_100_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or price crosses 4h EMA50
            if position == 1:
                # Exit long: RSI > 40 or price breaks below 4h EMA50
                if (rsi[i] > 40 or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: RSI < 60 or price breaks above 4h EMA50
                if (rsi[i] < 60 or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI14_Pullback_4hEMA50_1dEMA100_Volume"
timeframe = "1h"
leverage = 1.0