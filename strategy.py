#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d ATR-based breakout with weekly trend filter and volume confirmation
# Uses 1d price action with 1w trend context to avoid counter-trend trades
# Target: 8-20 trades/year, low frequency to minimize fee drag
# Works in bull markets via trend-following breakouts, in bear via volatility-filtered mean reversion
name = "1d_atr_breakout_1w_trend_volume_v1"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily ATR(14) for volatility and breakout threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily 20-period average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-day average
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA OR volatility drops significantly
            if close[i] <= ema_20_1w_aligned[i] or atr[i] < atr[i-1] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA OR volatility drops significantly
            if close[i] >= ema_20_1w_aligned[i] or atr[i] < atr[i-1] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above weekly EMA + 1*ATR + volume confirmation
            if close[i] > ema_20_1w_aligned[i] + atr[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly EMA - 1*ATR + volume confirmation
            elif close[i] < ema_20_1w_aligned[i] - atr[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals