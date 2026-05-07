#!/usr/bin/env python3
name = "1h_HTF_Direction_1h_Pullback_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for higher timeframe bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    for i in range(1, n):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i+1 > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i+1 > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # 4 hours cooldown
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction from both 4h and 1d
        trend_4h_up = close > ema_50_4h_aligned[i]
        trend_4h_down = close < ema_50_4h_aligned[i]
        trend_1d_up = close > ema_50_1d_aligned[i]
        trend_1d_down = close < ema_50_1d_aligned[i]
        
        # Require alignment between 4h and 1d trend
        trend_up = trend_4h_up and trend_1d_up
        trend_down = trend_4h_down and trend_1d_down
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Pullback in uptrend - RSI < 40 with volume
            if (rsi[i] < 40 and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Pullback in downtrend - RSI > 60 with volume
            elif (rsi[i] > 60 and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: RSI > 60 (overbought) or trend breaks down
            if rsi[i] > 60 or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI < 40 (oversold) or trend breaks up
            if rsi[i] < 40 or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h pullback strategy using 4h/1d trend alignment for direction.
# Enters on RSI extremes (<40 for long, >60 for short) during pullbacks in aligned trends.
# Uses volume confirmation to ensure momentum behind the move.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
# Holds through trend continuation, exits on mean reversion signals.