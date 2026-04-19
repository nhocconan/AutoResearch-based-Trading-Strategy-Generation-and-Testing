#!/usr/bin/env python3
# 4h_StochRSI_Trend_V1
# Hypothesis: 4h Stochastic RSI with trend filter and volume confirmation
# StochRSI identifies overbought/oversold conditions with momentum
# Combined with 4h EMA trend filter and volume spike to filter false signals
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in bull/bear via EMA trend filter - only long in uptrend, short in downtrend

name = "4h_StochRSI_Trend_V1"
timeframe = "4h"
leverage = 1.0

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
    
    # Stochastic RSI calculation
    def calculate_stochrsi(close_prices, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        # Calculate RSI
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for RSI
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        
        for i in range(rsi_period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic of RSI
        stoch_rsi = np.full_like(rsi, np.nan)
        for i in range(stoch_period-1, len(rsi)):
            if not np.isnan(rsi[i-stoch_period+1:i+1]).any():
                min_rsi = np.min(rsi[i-stoch_period+1:i+1])
                max_rsi = np.max(rsi[i-stoch_period+1:i+1])
                if max_rsi - min_rsi != 0:
                    stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
                else:
                    stoch_rsi[i] = 50
        
        # %K and %D
        k = np.full_like(stoch_rsi, np.nan)
        d = np.full_like(stoch_rsi, np.nan)
        
        for i in range(k_period-1, len(stoch_rsi)):
            if not np.isnan(stoch_rsi[i-k_period+1:i+1]).any():
                k[i] = np.mean(stoch_rsi[i-k_period+1:i+1])
        
        for i in range(d_period-1, len(k)):
            if not np.isnan(k[i-d_period+1:i+1]).any():
                d[i] = np.mean(k[i-d_period+1:i+1])
        
        return k, d  # Return %K and %D
    
    # 4h EMA for trend filter
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) >= period:
            multiplier = 2 / (period + 1)
            ema[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(ema[i-1]):
                    ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    # Calculate indicators on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for StochRSI and EMA
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # StochRSI on 4h
    stoch_k, stoch_d = calculate_stochrsi(close_4h, 14, 14, 3, 3)
    stoch_k_aligned = align_htf_to_ltf(prices, df_4h, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_4h, stoch_d)
    
    # EMA(50) for trend filter on 4h
    ema_50_4h = calculate_ema(close_4h, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(stoch_k_aligned[i]) or np.isnan(stoch_d_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA50 = uptrend, below = downtrend
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: StochRSI oversold (<20) and turning up, in uptrend with volume
            if (stoch_k_aligned[i] < 20 and 
                stoch_k_aligned[i] > stoch_d_aligned[i] and  # %K crossing above %D
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: StochRSI overbought (>80) and turning down, in downtrend with volume
            elif (stoch_k_aligned[i] > 80 and 
                  stoch_k_aligned[i] < stoch_d_aligned[i] and  # %K crossing below %D
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if StochRSI overbought (>80) or trend changes
            if (stoch_k_aligned[i] > 80) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if StochRSI oversold (<20) or trend changes
            if (stoch_k_aligned[i] < 20) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals