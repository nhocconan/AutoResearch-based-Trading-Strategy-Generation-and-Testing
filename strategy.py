# 12h_Wallace_Momentum_RSI_Trend
# Hypothesis: Use 12h RSI trend with 4h momentum confirmation and volume filter
# Works in bull markets via momentum continuation, works in bear via mean reversion at extremes
# Target: 15-30 trades/year on 12h timeframe with strong risk-adjusted returns

#!/usr/bin/env python3
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
    
    # Get 12h data for primary calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 14-period RSI on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate 20-period SMA for trend on 12h
    sma_20_12h = np.full(len(close_12h), np.nan)
    for i in range(19, len(close_12h)):
        sma_20_12h[i] = np.mean(close_12h[i-19:i+1])
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    vol_ma_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    vol_ratio_12h = np.divide(volume_12h, vol_ma_20_12h, out=np.full_like(volume_12h, np.nan), where=vol_ma_20_12h!=0)
    
    # Get 4h data for entry timing confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI for momentum confirmation
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.full(len(gain_4h), np.nan)
    avg_loss_4h = np.full(len(loss_4h), np.nan)
    
    for i in range(14, len(gain_4h)):
        if i == 14:
            avg_gain_4h[i] = np.mean(gain_4h[:14])
            avg_loss_4h[i] = np.mean(loss_4h[:14])
        else:
            avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
            avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    
    rs_4h = np.divide(avg_gain_4h, avg_loss_4h, out=np.full_like(avg_gain_4h, np.nan), where=avg_loss_4h!=0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    # Align all indicators to 12h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_20_12h)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup period
    start_idx = max(19, 14)  # Need SMA(20) and RSI(14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(sma_20_12h_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(rsi_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_12h = rsi_12h_aligned[i]
        sma_20 = sma_20_12h_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        rsi_4h = rsi_4h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold but recovering with volume confirmation and 4h momentum
            if (rsi_12h < 35 and rsi_12h > rsi_12h_aligned[i-1] and  # RSI rising from oversold
                vol_ratio > 1.5 and                                  # Volume confirmation
                rsi_4h > 50):                                        # 4h momentum bullish
                signals[i] = size
                position = 1
            # Short: RSI overbought but weakening with volume confirmation and 4h momentum
            elif (rsi_12h > 65 and rsi_12h < rsi_12h_aligned[i-1] and  # RSI falling from overbought
                  vol_ratio > 1.5 and                                  # Volume confirmation
                  rsi_4h < 50):                                        # 4h momentum bearish
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or momentum fails
            if rsi_12h > 70 or rsi_4h < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or momentum fails
            if rsi_12h < 30 or rsi_4h > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Wallace_Momentum_RSI_Trend"
timeframe = "12h"
leverage = 1.0