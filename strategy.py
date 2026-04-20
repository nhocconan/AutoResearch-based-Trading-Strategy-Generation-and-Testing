#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and daily data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend direction
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate daily RSI14 for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily volume SMA20 for volume filter
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Pre-compute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_34_4h_aligned[i]
        rsi_val = rsi_14_1d_aligned[i]
        vol_val = vol_sma_20_1d_aligned[i]
        vol_curr = df_1d['volume'].iloc[min(i // 24, len(df_1d)-1)] if i >= 24 else volume_1d[0]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(rsi_val) or 
            np.isnan(vol_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA34, RSI < 30 (oversold), volume above average
            if close_val > ema_val and rsi_val < 30 and vol_curr > vol_val:
                signals[i] = 0.20
                position = 1
            # Short: price below EMA34, RSI > 70 (overbought), volume above average
            elif close_val < ema_val and rsi_val > 70 and vol_curr > vol_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA34 or RSI > 70
            if close_val < ema_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above EMA34 or RSI < 30
            if close_val > ema_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# 4h_EMA34_1dRSI_VolumeFilter_V1
# Uses 4h EMA34 for trend direction, 1d RSI14 for overbought/oversold signals
# Enters long when price above EMA34, RSI < 30, and volume above average
# Enters short when price below EMA34, RSI > 70, and volume above average
# Exits on EMA34 cross or RSI reversal
# Uses session filter (08-20 UTC) to reduce noise
name = "4h_EMA34_1dRSI_VolumeFilter_V1"
timeframe = "1h"
leverage = 1.0