#!/usr/bin/env python3
"""
12h_rsi_pullback_1d_trend_volume_v1
Hypothesis: On 12-hour timeframe, enter long when RSI(14) pulls back to oversold (<30) during 1-day uptrend (price > EMA50) with volume confirmation; short when RSI overbought (>70) during 1-day downtrend (price < EMA50) with volume confirmation. Exit on RSI crossing 50. Designed for 12-30 trades/year to minimize fee drift while capturing mean-reversion within larger trends. Works in bull markets via pullbacks to uptrend and in bear markets via bounces off downtrend, using RSI extremes filtered by daily trend and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_pullback_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (price vs EMA50)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(close[i]):
            daily_trend_up[i] = close[i] > ema_50_1d_aligned[i]
            daily_trend_down[i] = close[i] < ema_50_1d_aligned[i]
    
    # Calculate RSI(14) on 12h timeframe
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(30, 50), n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: RSI pulls back to oversold (<30) during daily uptrend
                if (rsi[i] < 30 and rsi[i-1] >= 30 and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: RSI bounces from overbought (>70) during daily downtrend
                elif (rsi[i] > 70 and rsi[i-1] <= 70 and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals