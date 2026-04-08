#!/usr/bin/env python3
# 1d_ema200_rsi21_volume
# Hypothesis: On daily timeframe, enter long when price > EMA200 + RSI(21) < 30 + volume > 1.5x average; enter short when price < EMA200 + RSI(21) > 70 + volume > 1.5x average. Exit when price crosses back to EMA200 or RSI reverts to 50. Designed to capture mean-reversion within strong trends in both bull and bear markets. Low trade frequency (~10-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema200_rsi21_volume"
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
    
    # Get weekly data for trend filter (optional, but can add regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for regime filter (bull/bear)
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily EMA200 for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(21)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/21, adjust=False, min_periods=21).mean()
    avg_loss = loss.ewm(alpha=1/21, adjust=False, min_periods=21).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_200[i]) or np.isnan(rsi[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA200 OR RSI crosses above 50 (mean reversion)
            if close[i] < ema_200[i] or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA200 OR RSI crosses below 50 (mean reversion)
            if close[i] > ema_200[i] or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly regime filter: only long in bull market (price > weekly EMA200), only short in bear market (price < weekly EMA200)
            if close[i] > ema_200_1w_aligned[i]:
                # Bull market: look for long opportunities
                volume_ok = volume[i] > 1.5 * avg_volume[i]
                rsi_oversold = rsi[i] < 30
                price_above_ema = close[i] > ema_200[i]
                if price_above_ema and rsi_oversold and volume_ok:
                    position = 1
                    signals[i] = 0.25
            else:
                # Bear market: look for short opportunities
                volume_ok = volume[i] > 1.5 * avg_volume[i]
                rsi_overbought = rsi[i] > 70
                price_below_ema = close[i] < ema_200[i]
                if price_below_ema and rsi_overbought and volume_ok:
                    position = -1
                    signals[i] = -0.25
    
    return signals