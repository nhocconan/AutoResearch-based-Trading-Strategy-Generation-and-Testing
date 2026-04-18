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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 34-day EMA for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 14-day RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-day volume average
    vol_ma = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(20, len(volume_1d)):
            vol_ma[i] = np.mean(volume_1d[i-20:i])
    
    # Align all daily data to 4h timeframe
    upper_channel_4h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_4h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_4h = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_4h[i]) or np.isnan(lower_channel_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-day average
        vol_confirm = volume[i] > 1.8 * vol_ma_4h[i]
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema_34_4h[i]
        downtrend = close[i] < ema_34_4h[i]
        
        # RSI filter: avoid extremes
        rsi_ok = (rsi_4h[i] >= 35) and (rsi_4h[i] <= 65)
        
        if position == 0:
            # Long: break above upper channel with uptrend, volume, and RSI in range
            if close[i] > upper_channel_4h[i] and uptrend and vol_confirm and rsi_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with downtrend, volume, and RSI in range
            elif close[i] < lower_channel_4h[i] and downtrend and vol_confirm and rsi_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below lower channel OR trend reversal OR RSI overbought
            if (close[i] < lower_channel_4h[i]) or (not uptrend) or (rsi_4h[i] >= 70):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above upper channel OR trend reversal OR RSI oversold
            if (close[i] > upper_channel_4h[i]) or (not downtrend) or (rsi_4h[i] <= 30):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_RSI_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0