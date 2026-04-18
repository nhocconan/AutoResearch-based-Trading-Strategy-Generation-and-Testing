# %%
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
    
    # Get daily data for 50-period SMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate SMA50 on daily data
    close_1d_series = pd.Series(close_1d)
    sma50_1d = close_1d_series.rolling(window=50, min_periods=50).mean().values
    
    # Align daily SMA50 to 12h timeframe
    sma50_12h = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate 14-period RSI for momentum/overbought
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # need SMA50, RSI, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_12h[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price above daily SMA50, RSI not overbought, with volume
            if (close[i] > sma50_12h[i] and 
                rsi[i] < 60 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price below daily SMA50, RSI not oversold, with volume
            elif (close[i] < sma50_12h[i] and 
                  rsi[i] > 40 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below SMA50 or RSI overbought
            if close[i] < sma50_12h[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above SMA50 or RSI oversold
            if close[i] > sma50_12h[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_SMA50_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0
# %%