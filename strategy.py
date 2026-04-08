#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI + 1-week trend filter with volume confirmation
# RSI < 30 for long, > 70 for short on daily timeframe
# Trend filter: price above/below weekly EMA(34) to avoid counter-trend trades
# Volume confirmation: daily volume > 1.5x 20-day average
# Designed for low frequency (target: 15-25 trades/year) to avoid fee drag
# Works in both bull/bear markets via mean-reversion + trend filter
name = "daily_rsi_weekly_trend_volume_v1"
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
    
    # 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])   # First average of first 14 losses
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: 1d volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume_1d > (vol_ma * 1.5)
    
    # Align indicators to 1d timeframe (already aligned as we use daily data)
    rsi_aligned = rsi
    ema_34_aligned = ema_34_1w  # Weekly EMA values need to be aligned to daily
    vol_filter_aligned = vol_filter
    
    # Since we're using daily timeframe as primary, we need to align weekly data to daily
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or trend fails
            if rsi_aligned[i] > 50 or close[i] < ema_34_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or trend fails
            if rsi_aligned[i] < 50 or close[i] > ema_34_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34_aligned[i]
            bearish = close[i] < ema_34_aligned[i]
            
            # Long: RSI < 30 (oversold) + bullish trend + volume
            if (rsi_aligned[i] < 30 and 
                bullish and 
                vol_filter_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: RSI > 70 (overbought) + bearish trend + volume
            elif (rsi_aligned[i] > 70 and 
                  bearish and 
                  vol_filter_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals