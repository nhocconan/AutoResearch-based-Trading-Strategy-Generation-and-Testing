#!/usr/bin/env python3
name = "1d_RSI70_30_Pullback_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 1w EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 5  # ~1 week for 1d to reduce trades
    
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1w trend direction
        trend_up = close > ema_200_1w_aligned[i]
        trend_down = close < ema_200_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: RSI below 30 (oversold) in uptrend with volume
            if (rsi[i] < 30 and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: RSI above 70 (overbought) in downtrend with volume
            elif (rsi[i] > 70 and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: RSI crosses above 50 (mean reversion complete) or trend changes
            if rsi[i] > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 50 (mean reversion complete) or trend changes
            if rsi[i] < 50 or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI(14) mean reversion pullback strategy on 1d timeframe.
# Long when RSI < 30 (oversold) in 1w uptrend with volume confirmation.
# Short when RSI > 70 (overbought) in 1w downtrend with volume confirmation.
# Uses 1w EMA200 for trend filter to align with major trend.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
# Target: 30-100 total trades over 4 years (7-25/year) as per experiment guidelines.