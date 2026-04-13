#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    atr_period = 14
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Daily EMA for trend filter
    ema_period = 50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR-based volatility regime
    atr_ma_period = 50
    atr_ma = pd.Series(atr).rolling(window=atr_ma_period, min_periods=atr_ma_period).mean().values
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period MA (high volatility regime)
        vol_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        # Trend filter: only long when price > daily EMA50, short when price < daily EMA50
        long_trend = close[i] > ema_50_1d_aligned[i]
        short_trend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: breakout above/below ATR-based channels with volatility and trend
        # Upper channel: EMA50 + 1.5 * ATR
        upper_channel = ema_50_1d_aligned[i] + 1.5 * atr_aligned[i]
        # Lower channel: EMA50 - 1.5 * ATR
        lower_channel = ema_50_1d_aligned[i] - 1.5 * atr_aligned[i]
        
        breakout_up = close[i] > upper_channel
        breakout_down = close[i] < lower_channel
        
        if position == 0:
            if breakout_up and vol_filter and long_trend:
                position = 1
                signals[i] = position_size
            elif breakout_down and vol_filter and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to EMA50 or breaks below lower channel
            if close[i] <= ema_50_1d_aligned[i] or close[i] < lower_channel:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to EMA50 or breaks above upper channel
            if close[i] >= ema_50_1d_aligned[i] or close[i] > upper_channel:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ATR_Channel_Breakout_With_Volatility_Filter_v1"
timeframe = "4h"
leverage = 1.0