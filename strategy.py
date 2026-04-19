#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation.
# In 1h timeframe, RSI extremes often reverse, but only when aligned with higher timeframe trend.
# Use 4h EMA50 for trend direction: long when price > EMA50, short when price < EMA50.
# Enter when RSI(14) < 30 (oversold) in uptrend or > 70 (overbought) in downtrend.
# Require volume > 1.5x 20-period average to confirm momentum.
# Exit on opposite RSI extreme (RSI > 70 for longs, < 30 for shorts) or trend reversal.
# Session filter: only trade 08-20 UTC to avoid low-volume Asian session.
# Target: 20-40 trades/year by combining strict RSI extremes with trend/volume filters.

name = "1h_RSI_MeanReversion_TrendFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold in uptrend with volume surge
            if (close[i] > ema_50_4h_aligned[i] and 
                rsi[i] < 30 and 
                volume_surge[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in downtrend with volume surge
            elif (close[i] < ema_50_4h_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_surge[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if RSI overbought or trend turns down
            if (rsi[i] > 70) or (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if RSI oversold or trend turns up
            if (rsi[i] < 30) or (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals