#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily ATR Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: ATR breakouts capture volatility expansion moves; weekly trend filter ensures
# alignment with higher timeframe momentum; volume confirmation avoids false breakouts.
# Works in bull via breakouts with trend, in bear via mean-reversion at extremes during low volatility.
# Target: 10-25 trades/year to minimize fee drag on daily timeframe.
name = "1d_atr_breakout_1w_trend_volume_v2"
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
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR(14) for breakout threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily 20-period moving average for mean reversion reference
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily 20-period volume moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-day average volume
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price returns to 20-day MA OR weekly trend turns bearish
            if close[i] <= ma_20[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price returns to 20-day MA OR weekly trend turns bullish
            if close[i] >= ma_20[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Breakout threshold: 1.5 * ATR
            breakout_threshold = 1.5 * atr[i]
            
            # Enter long: price breaks above MA + ATR threshold + weekly uptrend + volume confirmation
            if (close[i] > ma_20[i] + breakout_threshold and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below MA - ATR threshold + weekly downtrend + volume confirmation
            elif (close[i] < ma_20[i] - breakout_threshold and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals