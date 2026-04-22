#!/usr/bin/env python3

"""
Hypothesis: 1-hour price action with 4-hour trend filter and daily volume confirmation.
Trades intraday pullbacks in the direction of higher timeframe trend using RSI extremes.
Uses daily volume spike to confirm institutional interest. Designed for low trade frequency
(15-35 trades/year) to minimize fee drag and work in both bull and bear markets by aligning
with higher timeframe trend and avoiding counter-trend trades.
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
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA for trend filter (20-period)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load daily data for volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 1-hour RSI (14-period) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x daily average
        vol_spike = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        if position == 0 and vol_spike:
            # Long: RSI oversold (<30) in uptrend (price > 4h EMA)
            if rsi_values[i] < 30 and close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) in downtrend (price < 4h EMA)
            elif rsi_values[i] > 70 and close[i] < ema_20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI > 50 or price below 4h EMA
                if rsi_values[i] > 50 or close[i] < ema_20_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI < 50 or price above 4h EMA
                if rsi_values[i] < 50 or close[i] > ema_20_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_Pullback_4hEMA20_DailyVol"
timeframe = "1h"
leverage = 1.0