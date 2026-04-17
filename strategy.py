#!/usr/bin/env python3
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
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 12h EMA5 for fast trend
    ema5_12h = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema5_12h_aligned = align_htf_to_ltf(prices, df_12h, ema5_12h)
    
    # 12h trend direction: EMA5 > EMA34
    trend_up = ema5_12h_aligned > ema34_12h_aligned
    trend_down = ema5_12h_aligned < ema34_12h_aligned
    
    # 4h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h volume filter
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema5_12h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: RSI < 40 (pullback in uptrend) with volume
            if (rsi[i] < 40 and trend_up[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (pullback in downtrend) with volume
            elif (rsi[i] > 60 and trend_down[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or trend change
            if (rsi[i] > 60 or not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or trend change
            if (rsi[i] < 40 or not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA5_EMA34_RSI_Pullback_Volume"
timeframe = "4h"
leverage = 1.0