#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_volume_v1"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily data
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_1d = rsi_14.values
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA(10,2,30) on weekly data
    close_1w_series = pd.Series(close_1w)
    change = abs(close_1w_series - close_1w_series.shift(10))
    volatility = abs(close_1w_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close_1w_series.iloc[0]]
    for i in range(1, len(close_1w_series)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1w_series.iloc[i] - kama[-1]))
    kama_1w = np.array(kama)
    
    # Align KAMA to 12h timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume filter - 20-period average on 12h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from KAMA
        uptrend = close[i] > kama_1w_aligned[i]
        downtrend = close[i] < kama_1w_aligned[i]
        
        # RSI signals with volume confirmation
        # Long: RSI < 30 (oversold) in uptrend
        long_signal = rsi_1d_aligned[i] < 30 and uptrend and volume_ok[i]
        # Short: RSI > 70 (overbought) in downtrend
        short_signal = rsi_1d_aligned[i] > 70 and downtrend and volume_ok[i]
        
        # Exit when RSI returns to neutral zone
        exit_long = rsi_1d_aligned[i] > 50
        exit_short = rsi_1d_aligned[i] < 50
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals