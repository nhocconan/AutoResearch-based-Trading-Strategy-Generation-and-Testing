#!/usr/bin/env python3
"""
1d_RSI_Divergence_WeeklyTrend_v1
Hypothesis: Daily RSI(14) divergences with weekly trend filter captures exhaustion moves in BTC/ETH. Bullish divergence (price LL, RSI HL) + weekly uptrend = long. Bearish divergence (price HH, RSI LH) + weekly downtrend = short. Uses volume confirmation (>1.5x avg) to filter weak signals. Designed for 1d to target 15-30 trades/year with discrete sizing (0.25).
"""

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
    
    # Load 1d data ONCE before loop for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter (more responsive than EMA50)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # RSI(14) on 1d close
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to 1d timeframe (wait for completed 1d bar)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Average volume for confirmation (24-period SMA = 1d * 0.4 = ~4h)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA(34), RSI(14), volume(24)
    start_idx = max(34, 14, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(rsi_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Trend filter: price vs weekly EMA34
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Need at least 2 periods to check for divergence
        if i < 2:
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
            
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = (close[i] < close[i-1] and close[i-1] < close[i-2] and 
                      rsi_val > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2])
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = (close[i] > close[i-1] and close[i-1] > close[i-2] and 
                      rsi_val < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2])
        
        # Long: bullish divergence + weekly uptrend + volume
        long_condition = bullish_div and uptrend and volume_confirmed
        # Short: bearish divergence + weekly downtrend + volume
        short_condition = bearish_div and downtrend and volume_confirmed
        
        # Exit: opposite divergence or loss of trend
        long_exit = (position == 1 and (bearish_div or not uptrend))
        short_exit = (position == -1 and (bullish_div or not downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_RSI_Divergence_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0