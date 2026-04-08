#!/usr/bin/env python3
# 1h_momentum_reversal_4h_trend_volume_v1
# Hypothesis: On 1h timeframe, enter short-term reversals when price deviates from 4h VWAP
# with volume confirmation, but only in direction of 4h trend (EMA50). This captures
# mean-reversion within stronger trends, working in both bull (buy dips) and bear (sell rallies).
# Uses 4h for trend direction and 1h for entry timing to limit trades to ~15-30/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_reversal_4h_trend_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend and VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h VWAP (typical price * volume / cumulative volume)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    vwap_4h = (typical_price_4h * volume_4h).cumsum() / volume_4h.cumsum()
    # Handle division by zero on first bar
    vwap_4h = np.where(volume_4h.cumsum() == 0, typical_price_4h, vwap_4h)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Calculate 1h RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume ratio (current vs 20-period average)
    avg_volume_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vwap_4h_aligned[i]) or np.isnan(avg_volume_1h[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (above 40) or price crosses above VWAP
            if rsi[i] > 40 or close[i] > vwap_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (below 60) or price crosses below VWAP
            if rsi[i] < 60 or close[i] < vwap_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume_1h[i]
            
            # Look for mean-reversion opportunities:
            # Long: RSI oversold (<30), price below 4h VWAP, but in 4h uptrend
            # Short: RSI overbought (>70), price above 4h VWAP, but in 4h downtrend
            if (rsi[i] < 30) and (close[i] < vwap_4h_aligned[i]) and \
               (close[i] > ema_50_4h_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.20
            elif (rsi[i] > 70) and (close[i] > vwap_4h_aligned[i]) and \
                 (close[i] < ema_50_4h_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.20
    
    return signals