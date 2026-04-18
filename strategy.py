#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 and volume calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA200
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to daily timeframe (1:1 mapping)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 50-day EMA for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily RSI(14)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.zeros_like(avg_gain_1d), where=avg_loss_1d!=0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily volume moving average (20-period)
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5 * 20-day average
        vol_confirmed = volume_1d[i] > 1.5 * vol_ma_1d[i] if i < len(volume_1d) else False
        
        if position == 0:
            # Long entry: price above EMA200, EMA50 trending up, RSI not overbought, with volume
            if (close[i] > ema200_aligned[i] and 
                ema50_aligned[i] > ema50_aligned[i-1] and 
                rsi_aligned[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price below EMA200, EMA50 trending down, RSI not oversold, with volume
            elif (close[i] < ema200_aligned[i] and 
                  ema50_aligned[i] < ema50_aligned[i-1] and 
                  rsi_aligned[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below EMA50 or RSI overbought
            if close[i] < ema50_aligned[i] or rsi_aligned[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA50 or RSI oversold
            if close[i] > ema50_aligned[i] or rsi_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA200_EMA50_RSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0