#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Stochastic RSI + 1d Supertrend filter + volume confirmation
# Stochastic RSI identifies overbought/oversold conditions with momentum.
# Supertrend from daily timeframe filters trend direction to avoid counter-trend trades.
# Volume spike confirms institutional participation.
# Target: 80-150 total trades over 4 years with controlled risk in all market regimes.

name = "6h_stochrsi_1d_supertrend_vol_v1"
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
    
    # 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend calculation (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + multiplier * atr_1d
    basic_lb = (high_1d + low_1d) / 2 - multiplier * atr_1d
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_1d))
    final_lb = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(len(close_1d))
    trend = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1d)):
        if i == 0:
            supertrend[i] = final_ub[i]
            trend[i] = 1
        else:
            if trend[i-1] == 1:
                if close_1d[i] <= final_ub[i-1]:
                    supertrend[i] = final_ub[i]
                    trend[i] = 1
                else:
                    supertrend[i] = final_lb[i]
                    trend[i] = -1
            else:
                if close_1d[i] >= final_lb[i-1]:
                    supertrend[i] = final_lb[i]
                    trend[i] = -1
                else:
                    supertrend[i] = final_ub[i]
                    trend[i] = 1
    
    # Align 1d Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Stochastic RSI calculation (14,14,3,3)
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).max().values
    
    # Avoid division by zero
    rsi_range = rsi_max - rsi_min
    stoch_rsi = np.divide((rsi - rsi_min) * 100, rsi_range, out=np.zeros_like(rsi), where=rsi_range!=0)
    
    # %K and %D
    k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean().values
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup periods
        # Skip if required data not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Supertrend turns bearish or StochRSI overbought
            elif trend_aligned[i] == -1 or k[i] > 80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Supertrend turns bullish or StochRSI oversold
            elif trend_aligned[i] == 1 or k[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: Supertrend uptrend, StochRSI oversold (<20) and crossing up, volume spike
            if (trend_aligned[i] == 1 and 
                k[i] < 20 and 
                k[i] > d[i] and  # bullish crossover
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Supertrend downtrend, StochRSI overbought (>80) and crossing down, volume spike
            elif (trend_aligned[i] == -1 and 
                  k[i] > 80 and 
                  k[i] < d[i] and  # bearish crossover
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals