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
    
    # Get daily data for weekly EMA200 and weekly RSI(14)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 (trend filter)
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly RSI(14)
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.zeros_like(avg_gain_1w), where=avg_loss_1w!=0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Align weekly indicators to 4h timeframe
    ema200_4h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    rsi_1w_4h = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 4h RSI(14) for momentum
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
    
    start_idx = max(200, 20)  # need weekly EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_4h[i]) or np.isnan(rsi_1w_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price above weekly EMA200, weekly RSI not oversold, RSI not overbought, with volume
            if (close[i] > ema200_4h[i] and 
                rsi_1w_4h[i] > 30 and 
                rsi[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly EMA200, weekly RSI not overbought, RSI not oversold, with volume
            elif (close[i] < ema200_4h[i] and 
                  rsi_1w_4h[i] < 70 and 
                  rsi[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA200 or RSI overbought
            if close[i] < ema200_4h[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA200 or RSI oversold
            if close[i] > ema200_4h[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyEMA200_WeeklyRSI_RSI_Volume_Filter"
timeframe = "4h"
leverage = 1.0