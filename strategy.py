#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly RSI with 12h Trend Filter and Volume Confirmation
# Hypothesis: Weekly RSI extremes (overbought/oversold) combined with 12h trend direction
# and volume spikes provide high-probability mean-reversion entries in both bull and bear markets.
# Uses 12h EMA50 as trend filter to align with higher timeframe momentum.
# Volume > 2x 20-period average confirms institutional participation.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_weekly_rsi_12h_trend_volume_v1"
timeframe = "6h"
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
    
    # Get weekly data for RSI calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    weekly_rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 6h timeframe
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_rsi_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or trend turns bearish or volume drops
            if (weekly_rsi_aligned[i] >= 50 or close[i] < ema_50_12h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or trend turns bullish or volume drops
            if (weekly_rsi_aligned[i] <= 50 or close[i] > ema_50_12h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: weekly RSI oversold (<30) with bullish 12h trend and volume spike
            if (weekly_rsi_aligned[i] < 30 and close[i] > ema_50_12h_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly RSI overbought (>70) with bearish 12h trend and volume spike
            elif (weekly_rsi_aligned[i] > 70 and close[i] < ema_50_12h_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals