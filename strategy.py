# 4h_RSI_Stochastic_Trend_Follower
# Hypothesis: 4h timeframe strategy using RSI(14) trend filter + Stochastic(14,3,3) momentum + volume confirmation
# RSI > 50 for long, < 50 for short filters for trend direction
# Stochastic identifies overbought/oversold conditions for entry timing in trending markets
# Volume confirmation ensures institutional participation
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in bull/bear via RSI trend filter and volatility-adjusted entries

name = "4h_RSI_Stochastic_Trend_Follower"
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
    
    # RSI(14) for trend filter
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        # Initial average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Stochastic Oscillator (14,3,3)
    def calculate_stochastic(high_prices, low_prices, close_prices, k_period=14, d_period=3):
        # %K calculation
        lowest_low = np.zeros_like(low_prices)
        highest_high = np.zeros_like(high_prices)
        
        for i in range(len(low_prices)):
            if i < k_period - 1:
                lowest_low[i] = np.nan
                highest_high[i] = np.nan
            else:
                lowest_low[i] = np.min(low_prices[i-k_period+1:i+1])
                highest_high[i] = np.max(high_prices[i-k_period+1:i+1])
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        k_percent = np.where(denominator != 0, 
                            100 * (close_prices - lowest_low) / denominator, 
                            50)
        
        # %D calculation (SMA of %K)
        d_percent = np.full_like(k_percent, np.nan)
        for i in range(len(k_percent)):
            if i < d_period - 1:
                d_percent[i] = np.nan
            else:
                valid_k = k_percent[i-d_period+1:i+1]
                if not np.any(np.isnan(valid_k)):
                    d_percent[i] = np.mean(valid_k)
                else:
                    d_percent[i] = np.nan
        
        return k_percent, d_percent
    
    # 4h data for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate RSI on 4h data
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate Stochastic on 4h data
    stoch_k, stoch_d = calculate_stochastic(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14, 3
    )
    stoch_k_aligned = align_htf_to_ltf(prices, df_4h, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_4h, stoch_d)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(stoch_k_aligned[i]) or 
            np.isnan(stoch_d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: RSI > 50 for long bias, < 50 for short bias
        long_bias = rsi_4h_aligned[i] > 50
        short_bias = rsi_4h_aligned[i] < 50
        
        if position == 0:
            # Long: RSI > 50 (uptrend) + Stochastic oversold (< 20) + volume
            if (long_bias and 
                stoch_k_aligned[i] < 20 and 
                stoch_d_aligned[i] < 20 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 (downtrend) + Stochastic overbought (> 80) + volume
            elif (short_bias and 
                  stoch_k_aligned[i] > 80 and 
                  stoch_d_aligned[i] > 80 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI < 40 (trend weakening) or Stochastic overbought (> 80)
            if (rsi_4h_aligned[i] < 40) or (stoch_k_aligned[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI > 60 (trend weakening) or Stochastic oversold (< 20)
            if (rsi_4h_aligned[i] > 60) or (stoch_k_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals