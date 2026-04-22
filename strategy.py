# 1H_Momentum_4hTrend_1dVolume - 1h momentum aligned with 4h trend and daily volume
# Uses 4h EMA for trend direction, 1h RSI for momentum entry, and 1d volume filter
# Designed for low trade frequency (15-35/year) with momentum + trend alignment
# Works in both bull and bear markets by following higher timeframe trend

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
    
    # Load 4h data for trend - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA (20-period) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d average volume (20-period) for volume filter
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 1h RSI (14-period) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(rsi[i])):
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
            # Long: 4h uptrend + 1h momentum (RSI > 50) + volume above average
            if (close[i] > ema_4h_aligned[i] and  # Price above 4h EMA (uptrend)
                rsi[i] > 50 and                 # Bullish momentum
                volume[i] > avg_vol_1d_aligned[i]):  # Above average daily volume
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1h momentum (RSI < 50) + volume above average
            elif (close[i] < ema_4h_aligned[i] and  # Price below 4h EMA (downtrend)
                  rsi[i] < 50 and                 # Bearish momentum
                  volume[i] > avg_vol_1d_aligned[i]):  # Above average daily volume
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: trend reversal or momentum exhaustion
            exit_signal = False
            
            if position == 1:
                # Exit long: price below 4h EMA OR RSI < 40 (momentum loss)
                if close[i] < ema_4h_aligned[i] or rsi[i] < 40:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above 4h EMA OR RSI > 60 (momentum loss)
                if close[i] > ema_4h_aligned[i] or rsi[i] > 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Momentum_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0
#%%