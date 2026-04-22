#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
Long when RSI < 30 (oversold) in a 4h uptrend (close > EMA50) with volume spike.
Short when RSI > 70 (overbought) in a 4h downtrend (close < EMA50) with volume spike.
Exit when RSI crosses 50 (mean reversion complete) or trend weakens.
Uses 4h EMA for trend filter and volume spike for entry confirmation.
Designed for low trade frequency (15-35/year) to minimize fee drag.
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 50-period EMA on 4h close
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 20-period volume average on 1h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold in 4h uptrend with volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_4h_aligned[i] and  # 4h uptrend
                volume[i] > 2.0 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in 4h downtrend with volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_4h_aligned[i] and  # 4h downtrend
                  volume[i] > 2.0 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses above 50 OR 4h trend turns down
                if rsi[i] > 50 or close[i] < ema_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses below 50 OR 4h trend turns up
                if rsi[i] < 50 or close[i] > ema_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_RSI_MeanReversion_4hEMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0
#%%