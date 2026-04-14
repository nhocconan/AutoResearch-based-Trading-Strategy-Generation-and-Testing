#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) - Wilder's smoothing
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(df_1d), np.nan)
    avg_loss = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(df_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align indicators to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 12-hour price range (high-low) for volatility filter
    price_range = high - low
    range_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            range_ma[i] = np.mean(price_range[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or
            np.isnan(rsi_12h[i]) or
            np.isnan(range_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (range < 0.5% of price)
        if range_ma[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip extreme RSI (overbought/oversold) - mean reversion in range
        if rsi_12h[i] > 70 or rsi_12h[i] < 30:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price near daily low AND RSI < 40 (oversold bounce)
            if low[i] <= low_1d[i] * 1.005 and rsi_12h[i] < 40:
                position = 1
                signals[i] = position_size
            # Short: Price near daily high AND RSI > 60 (overbought rejection)
            elif high[i] >= high_1d[i] * 0.995 and rsi_12h[i] > 60:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price reaches daily high OR RSI > 60 (overbought)
            if high[i] >= high_1d[i] * 0.995 or rsi_12h[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price reaches daily low OR RSI < 40 (oversold)
            if low[i] <= low_1d[i] * 1.005 or rsi_12h[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_RSI_Range_MeanReversion"
timeframe = "12h"
leverage = 1.0
#%%