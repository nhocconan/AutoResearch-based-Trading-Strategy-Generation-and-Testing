#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily 20-period EMA for trend
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily 14-period ATR for volatility filter and stop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily 20-period volume moving average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4-period RSI for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4 = 100 - (100 / (1 + rs))
    rsi_4_aligned = align_htf_to_ltf(prices, df_1d, rsi_4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(rsi_4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price above daily EMA20, bullish momentum (RSI > 50), volume confirmation
            if (price > ema_20_1d_aligned[i] and 
                rsi_4_aligned[i] > 50 and 
                vol > 1.3 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA20, bearish momentum (RSI < 50), volume confirmation
            elif (price < ema_20_1d_aligned[i] and 
                  rsi_4_aligned[i] < 50 and 
                  vol > 1.3 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA20 or momentum turns bearish (RSI < 40)
            if price < ema_20_1d_aligned[i] or rsi_4_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA20 or momentum turns bullish (RSI > 60)
            if price > ema_20_1d_aligned[i] or rsi_4_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA20_RSI4_VolumeFilter"
timeframe = "1d"
leverage = 1.0