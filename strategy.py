#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Bollinger Bands (20, 2.5) - using close
    close_1d_series = pd.Series(close_1d)
    ma20 = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_1d_series.rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + 2.5 * std20
    lower_bb = ma20 - 2.5 * std20
    
    # 1d Average True Range (14) for volatility
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_1d[0] = high_low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d RSI (14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    ma20_aligned = align_htf_to_ltf(prices, df_1d, ma20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h volume spike detection (20-period average)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 14)
    for i in range(start, n):
        if (np.isnan(ma20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        ma20_val = ma20_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        atr_val = atr_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price touches lower BB + RSI oversold + volume spike
            if (price <= lower_bb_val and rsi_val < 30 and vol > 1.5 * vol_ma_val):
                position = 1
                signals[i] = position_size
            # Short: price touches upper BB + RSI overbought + volume spike
            elif (price >= upper_bb_val and rsi_val > 70 and vol > 1.5 * vol_ma_val):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above middle band OR RSI overbought
            if (price >= ma20_val or rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below middle band OR RSI oversold
            if (price <= ma20_val or rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_RSI_Volume_Spike"
timeframe = "4h"
leverage = 1.0