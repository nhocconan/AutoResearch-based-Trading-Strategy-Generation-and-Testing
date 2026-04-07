#!/usr/bin/env python3
"""
1d_rsi_pullback_1w_trend_volume_v1
Hypothesis: On daily timeframe, use weekly RSI to identify pullbacks in the prevailing weekly trend. Enter long when daily RSI < 30 and weekly RSI > 50 in uptrend, short when daily RSI > 70 and weekly RSI < 50 in downtrend. Volume confirmation filters low-volatility noise. Designed for 15-25 trades/year to minimize fee drag while capturing trend resumption after pullbacks in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_pullback_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[:period-1])
        avg_loss[period-1] = np.mean(loss[:period-1])
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period-1] = np.nan
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    
    # Calculate weekly RSI for trend filter
    w_rsi = calculate_rsi(w_close, 14)
    
    # Calculate daily RSI for entry signals
    d_rsi = calculate_rsi(close, 14)
    
    # Volume filter: daily volume > 1.3x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    # Align weekly RSI to daily timeframe
    w_rsi_aligned = align_htf_to_ltf(prices, df_1w, w_rsi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if weekly RSI not available
        if np.isnan(w_rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if daily RSI not available
        if np.isnan(d_rsi[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend based on RSI
        weekly_uptrend = w_rsi_aligned[i] > 50
        weekly_downtrend = w_rsi_aligned[i] < 50
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.3
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when daily RSI reaches overbought (take profit)
            if d_rsi[i] >= 70:
                exit_long = True
            # Exit when weekly trend turns down
            elif not weekly_uptrend:
                exit_long = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.7:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when daily RSI reaches oversold (take profit)
            if d_rsi[i] <= 30:
                exit_short = True
            # Exit when weekly trend turns up
            elif not weekly_downtrend:
                exit_short = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.7:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: daily RSI oversold in weekly uptrend with volume
            long_entry = (d_rsi[i] < 30) and weekly_uptrend and vol_confirmed
            
            # Short entry: daily RSI overbought in weekly downtrend with volume
            short_entry = (d_rsi[i] > 70) and weekly_downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals