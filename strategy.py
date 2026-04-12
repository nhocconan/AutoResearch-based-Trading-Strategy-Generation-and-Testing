#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_chaikin_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chaikin Money Flow calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Chaikin Money Flow (CMF) for daily timeframe
    # CMF = Sum of Money Flow Volume over N periods / Sum of Volume over N periods
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mf_multiplier = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if high_1d[i] != low_1d[i]:
            mf_multiplier[i] = ((close_1d[i] - low_1d[i]) - (high_1d[i] - close_1d[i])) / (high_1d[i] - low_1d[i])
        else:
            mf_multiplier[i] = 0.0
    
    mf_volume = mf_multiplier * volume_1d
    
    # 20-period CMF
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = np.divide(mf_volume_sum, volume_sum, out=np.zeros_like(mf_volume_sum), where=volume_sum!=0)
    
    # Align CMF to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    
    # Calculate 4h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h volume filter
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    position_size = 0.25  # Fixed position size
    
    for i in range(30, n):  # warmup
        # Skip if not ready
        if np.isnan(cmf_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i]):
            signals[i] = 0.0 if position == 0 else (position_size if position == 1 else -position_size)
            continue
        
        # Entry conditions
        # Long: CMF > 0.1 (buying pressure) AND RSI < 60 (not overbought) AND volume confirmation
        long_signal = cmf_aligned[i] > 0.1 and rsi[i] < 60 and volume_ok[i]
        # Short: CMF < -0.1 (selling pressure) AND RSI > 40 (not oversold) AND volume confirmation
        short_signal = cmf_aligned[i] < -0.1 and rsi[i] > 40 and volume_ok[i]
        
        # Exit conditions
        # Exit long when CMF turns negative or RSI > 70
        exit_long = cmf_aligned[i] < 0 or rsi[i] > 70
        # Exit short when CMF turns positive or RSI < 30
        exit_short = cmf_aligned[i] > 0 or rsi[i] < 30
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals