#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_v2
Hypothesis: Daily breakouts at Camarilla R1/S1 levels with weekly trend filter and volume confirmation. 
This version tightens entry conditions by requiring volume > 2.5x average (reducing false breakouts) 
and uses Wilder's RSI(14) < 40 for long and > 60 for short to avoid chasing extremes. 
Target: 15-25 trades/year. Works in bull/bear markets by trading with weekly trend.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # RSI(14) for momentum filter
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Align all higher timeframe data to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Trend filter: price > EMA34 = bullish, < EMA34 = bearish
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Volume confirmation: current volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_surge[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment, volume surge, and RSI filter
        # Long: price breaks above Camarilla R1 + weekly uptrend + volume surge + RSI < 40
        long_entry = (close[i] > camarilla_r1_aligned[i] and 
                     trend_up[i] and 
                     volume_surge[i] and 
                     rsi_values[i] < 40)
        
        # Short: price breaks below Camarilla S1 + weekly downtrend + volume surge + RSI > 60
        short_entry = (close[i] < camarilla_s1_aligned[i] and 
                      trend_down[i] and 
                      volume_surge[i] and 
                      rsi_values[i] > 60)
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < camarilla_s1_aligned[i] and volume_surge[i]
        short_exit = close[i] > camarilla_r1_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_v2"
timeframe = "1d"
leverage = 1.0