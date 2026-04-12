#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_rsi_divergence_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily data
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter - 20-period average on 12h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # RSI divergence detection
    price_highs = pd.Series(high).rolling(window=5, center=True).max().values
    price_lows = pd.Series(low).rolling(window=5, center=True).min().values
    
    # For divergence: look for price making new high/low while RSI doesn't
    rsi_series = pd.Series(rsi_1d_aligned)
    rsi_highs = rsi_series.rolling(window=5, center=True).max().values
    rsi_lows = rsi_series.rolling(window=5, center=True).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from 1w EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 10:
            if (low[i] < low[i-5] and rsi_1d_aligned[i] > rsi_1d_aligned[i-5] and
                rsi_1d_aligned[i] < 30 and uptrend):
                bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 10:
            if (high[i] > high[i-5] and rsi_1d_aligned[i] < rsi_1d_aligned[i-5] and
                rsi_1d_aligned[i] > 70 and downtrend):
                bearish_div = True
        
        # Volume confirmation
        vol_confirm = volume_ok[i]
        
        # Execute trades
        if bullish_div and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_div and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold or close
            if position == 1 and (rsi_1d_aligned[i] > 70 or not uptrend):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (rsi_1d_aligned[i] < 30 or not downtrend):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals