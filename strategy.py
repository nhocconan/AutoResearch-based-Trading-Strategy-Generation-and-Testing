#!/usr/bin/env python3
# daily_rsi20_reversal_v1
# Hypothesis: Daily RSI(14) extremes (RSI<30 for long, RSI>70 for short) with volume confirmation (>1.5x 20-day avg) and weekly trend filter (price above/below weekly EMA50) work in both bull and bear markets by capturing mean reversion with trend alignment.
# Target: 15-25 trades/year (60-100 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_rsi20_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Weekly trend filter: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or weekly trend turns bearish
            if rsi[i] > 50 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or weekly trend turns bullish
            if rsi[i] < 50 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI < 30 (oversold) with volume confirmation and weekly uptrend
            if rsi[i] < 30 and volume[i] > vol_threshold[i] and close[i] > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI > 70 (overbought) with volume confirmation and weekly downtrend
            elif rsi[i] > 70 and volume[i] > vol_threshold[i] and close[i] < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals