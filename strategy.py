#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 20-period ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR20 on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20d = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # Align daily ATR20 to 4h timeframe
    atr_20d_aligned = align_htf_to_ltf(prices, df_1d, atr_20d)
    
    # Calculate 4-period RSI on close (momentum)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_4 = 100 - (100 / (1 + rs))
    
    # Calculate 4h ATR for stop loss
    tr_4h_1 = high - low
    tr_4h_2 = np.abs(high - np.roll(close, 1))
    tr_4h_3 = np.abs(low - np.roll(close, 1))
    tr_4h_1[0] = high[0] - low[0]
    tr_4h_2[0] = np.abs(high[0] - close[0])
    tr_4h_3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h_1, np.maximum(tr_4h_2, tr_4h_3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need ATR20d and ATR4h
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(atr_20d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(rsi_4[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR20d > 0.8 * its 20-period average (avoid low volatility)
        if i >= 40:
            atr_ma = np.mean(atr_20d_aligned[i-20:i])
            vol_filter = atr_20d_aligned[i] > 0.8 * atr_ma
        else:
            vol_filter = True  # not enough data for MA, allow
        
        # Momentum filter: RSI(4) between 20 and 80 (avoid extremes)
        mom_filter = (rsi_4[i] >= 20) and (rsi_4[i] <= 80)
        
        if position == 0:
            # Long entry: close above open + 0.3*ATR4h, with filters
            if (close[i] > open_prices[i] + 0.3 * atr_4h[i] and 
                vol_filter and 
                mom_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: close below open - 0.3*ATR4h, with filters
            elif (close[i] < open_prices[i] - 0.3 * atr_4h[i] and 
                  vol_filter and 
                  mom_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below open - 1.0*ATR4h (stop) or RSI > 70 (take profit)
            if close[i] < open_prices[i] - 1.0 * atr_4h[i] or rsi_4[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above open + 1.0*ATR4h (stop) or RSI < 30 (take profit)
            if close[i] > open_prices[i] + 1.0 * atr_4h[i] or rsi_4[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ATR20D_VolFilter_RSI4_Momentum"
timeframe = "4h"
leverage = 1.0