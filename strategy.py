# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy using 4h RSI trend filter and 1d volume regime filter
# Uses RSI(14) on 4h for trend direction (RSI > 50 = bullish, < 50 = bearish)
# Entry on 1h when price crosses EMA(20) with volume above 1.5x average
# Volume regime filter: only trade when 1d volume is above its 20-period average (high activity days)
# Designed to work in both bull and bear markets by following the 4h RSI trend
# Target: 60-150 total trades over 4 years (15-37/year) with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA(20) on 1h
    ema_period = 20
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Calculate RSI(14) on 4h
    rsi_period = 14
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average on 1d
    vol_ma_period = 20
    vol_ma = pd.Series(volume_1d).rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    
    # Align indicators to 1h timeframe
    ema_aligned = ema  # already on 1h
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size (20%)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            continue
        
        # Volume regime filter: only trade when 1d volume is above average
        volume_regime = volume_1d[i // 24] > vol_ma[i // 24] if i // 24 < len(volume_1d) else False
        
        # Long entry: price crosses above EMA(20) + 4h RSI > 50 (bullish) + volume regime
        if (close[i] > ema_aligned[i] and 
            rsi_aligned[i] > 50 and 
            volume_regime and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price crosses below EMA(20) + 4h RSI < 50 (bearish) + volume regime
        elif (close[i] < ema_aligned[i] and 
              rsi_aligned[i] < 50 and 
              volume_regime and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse EMA cross or 4h RSI crosses 50 (trend change)
        elif position == 1 and (close[i] < ema_aligned[i] or rsi_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_aligned[i] or rsi_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSITrend_VolumeRegime_EMA_Cross"
timeframe = "1h"
leverage = 1.0