#!/usr/bin/env python3
"""
Hypothesis: Daily RSI reversal with weekly trend filter and volume confirmation.
In bull market (weekly close > weekly EMA10): look for long on RSI(14) < 30 reversal.
In bear market (weekly close < weekly EMA10): look for short on RSI(14) > 70 reversal.
Volume must be above 20-day average to confirm reversal strength.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_rsi_reversal_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=10, adjust=False, min_periods=10).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)  # already shifted
    
    # === DAILY RSI (14) ===
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(rsi_period, n):
        if np.isnan(one_w_ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        bull_trend = close[i] > one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) OR trend turns bearish
            if rsi_values[i] > 70 or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) OR trend turns bullish
            if rsi_values[i] < 30 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend
            if bull_trend:
                # In bull market: long on RSI < 30 (oversold bounce)
                if rsi_values[i] < 30 and rsi_values[i-1] >= 30:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on RSI > 70 (overbought rejection)
                if rsi_values[i] > 70 and rsi_values[i-1] <= 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals