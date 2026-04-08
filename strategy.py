#!/usr/bin/env python3
# 1d_rsi20_reversion_1w_trend_volume
# Hypothesis: Daily RSI(20) mean reversion with weekly EMA50 trend filter and volume confirmation.
# Long when RSI<30 and price > weekly EMA50 and volume > 1.5x average.
# Short when RSI>70 and price < weekly EMA50 and volume > 1.5x average.
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# Designed to capture oversold bounces in uptrends and overbought pullbacks in downtrends.
# Target: 30-80 total trades over 4 years (~7-20/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi20_reversion_1w_trend_volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(20)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    avg_loss = loss.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses back above 50
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses back below 50
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Mean reversion entries: RSI oversold (long) and overbought (short)
            if (rsi[i] < 30) and (close[i] > ema_50_1w_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (rsi[i] > 70) and (close[i] < ema_50_1w_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals