#!/usr/bin/env python3
# 1d_weekly_rsi_reversion_v1
# Hypothesis: RSI mean reversion on daily timeframe with weekly trend filter and volume confirmation.
# Long when: RSI(14) < 30 (oversold), weekly RSI(14) > 50 (bullish trend), volume > 1.5x average.
# Short when: RSI(14) > 70 (overbought), weekly RSI(14) < 50 (bearish trend), volume > 1.5x average.
# Exit when RSI returns to neutral (40-60 range) or volume drops below average.
# Uses weekly trend to avoid counter-trend trades in strong markets.
# Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily RSI(14) for mean reversion signals
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate RSI using Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly RSI(14) for trend direction
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.zeros(len(close_1w))
    avg_loss_1w = np.zeros(len(close_1w))
    if len(close_1w) > rsi_period:
        avg_gain_1w[rsi_period] = np.mean(gain_1w[1:rsi_period+1])
        avg_loss_1w[rsi_period] = np.mean(loss_1w[1:rsi_period+1])
        
        for i in range(rsi_period+1, len(close_1w)):
            avg_gain_1w[i] = (avg_gain_1w[i-1] * (rsi_period-1) + gain_1w[i]) / rsi_period
            avg_loss_1w[i] = (avg_loss_1w[i-1] * (rsi_period-1) + loss_1w[i]) / rsi_period
    
    rs_1w = np.where(avg_loss_1w != 0, avg_gain_1w / avg_loss_1w, 0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(rsi_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or volume drops
            if rsi[i] >= 40 or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or volume drops
            if rsi[i] <= 60 or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Daily RSI oversold (<30), weekly RSI bullish (>50), volume surge
            if (rsi[i] < 30 and 
                rsi_1w_aligned[i] > 50 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Daily RSI overbought (>70), weekly RSI bearish (<50), volume surge
            elif (rsi[i] > 70 and 
                  rsi_1w_aligned[i] < 50 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals