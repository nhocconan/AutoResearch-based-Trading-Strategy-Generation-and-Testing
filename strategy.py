#!/usr/bin/env python3
# Hypothesis: 6h Stochastic RSI + 1d Williams %R for mean reversion in oversold/overbought zones
# Long when StochRSI < 0.2, Williams %R < -80 (extreme oversold), and price > 6h EMA50 (trend filter)
# Short when StochRSI > 0.8, Williams %R > -20 (extreme overbought), and price < 6h EMA50
# Exit when StochRSI crosses above 0.5 (long) or below 0.5 (short)
# Uses mean reversion with trend alignment to avoid counter-trend trades, targeting 60-120 trades over 4 years
# Stochastic RSI identifies momentum extremes, Williams %R confirms overbought/oversold, EMA50 filters trend

name = "6h_StochRSI_WilliamsR_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 6h Stochastic RSI
    rsi_period = 14
    stoch_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi = stoch_rsi.fillna(0.5)
    stoch_rsi_values = stoch_rsi.values
    
    # Calculate 1d Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.fillna(-50)
    williams_r_values = williams_r.values
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
    
    # Calculate 6h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(stoch_rsi_values[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: extreme oversold with uptrend filter
            if (stoch_rsi_values[i] < 0.2 and 
                williams_r_aligned[i] < -80 and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: extreme overbought with downtrend filter
            elif (stoch_rsi_values[i] > 0.8 and 
                  williams_r_aligned[i] > -20 and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: StochRSI crosses above 0.5 (mean reversion complete)
            if stoch_rsi_values[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: StochRSI crosses below 0.5 (mean reversion complete)
            if stoch_rsi_values[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals