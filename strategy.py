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
    
    # Get daily data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 4-hour Bollinger Bands (20-period SMA, 2 std)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for SMA20, std20, vol MA, and daily indicators
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        atr_vol = atr14_1d_aligned[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        vol_spike_val = vol_spike[i]
        
        # Dynamic Bollinger width threshold: avoid trading in extremely low volatility
        bb_width = (upper_bb_val - lower_bb_val) / sma20[i] if sma20[i] != 0 else 0
        min_bb_width = 0.01  # 1% of price
        
        if position == 0:
            # Long: price touches lower BB + volume spike + uptrend + sufficient volatility
            if (close[i] <= lower_bb_val and vol_spike_val and 
                close[i] > ema_trend and bb_width > min_bb_width):
                signals[i] = size
                position = 1
            # Short: price touches upper BB + volume spike + downtrend + sufficient volatility
            elif (close[i] >= upper_bb_val and vol_spike_val and 
                  close[i] < ema_trend and bb_width > min_bb_width):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA20 or trend turns down
            if close[i] >= sma20[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below SMA20 or trend turns up
            if close[i] <= sma20[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Bounce_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0