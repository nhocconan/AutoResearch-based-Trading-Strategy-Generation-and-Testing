#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-week EMA trend filter and 1-day RSI mean-reversion with volume confirmation.
- Long when price > weekly EMA50 (trend up) + daily RSI < 40 (oversold) + volume > 1.5x 20-period volume MA
- Short when price < weekly EMA50 (trend down) + daily RSI > 60 (overbought) + volume > 1.5x 20-period volume MA
- Exit when RSI returns to neutral (40-60 range) or opposite signal triggers
- Fixed position size 0.25 to manage drawdown
- Uses weekly trend for direction and daily RSI for timing, reducing whipsaw in ranging markets
- Designed for 12h timeframe with strict entry conditions to target 50-150 total trades over 4 years
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14) for mean-reversion signals
    df_1d = get_htf_data(prices, '1d')
    rsi_period = 14
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for weekly EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        ema_trend = ema_50_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Look for RSI extremes with volume confirmation and weekly trend filter
            # Long: price above weekly EMA50 + RSI oversold (<40) + volume spike
            if price > ema_trend and rsi_val < 40 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA50 + RSI overbought (>60) + volume spike
            elif price < ema_trend and rsi_val > 60 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when RSI returns to neutral (40-60) or opposite condition
            if rsi_val >= 40:  # RSI recovered from oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when RSI returns to neutral (40-60) or opposite condition
            if rsi_val <= 60:  # RSI dropped from overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA50_DailyRSI_Volume"
timeframe = "12h"
leverage = 1.0