#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI with 4h Trend Filter and 1d Volume Confirmation
# Takes long when 1h RSI crosses above 40 with 4h EMA21 uptrend and 1d volume above average
# Takes short when 1h RSI crosses below 60 with 4h EMA21 downtrend and 1d volume above average
# Uses RSI for mean reversion entries aligned with higher timeframe trend
# Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag
# Works in both bull and bear: RSI mean reversion works in ranges, trend filter avoids counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate 4h EMA21 for trend
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 30  # for RSI and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        rsi_val = rsi[i]
        ema_4h_val = ema_4h_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        vol_1d_current = volume_1d[i] if i < len(volume_1d) else volume_1d[-1]
        
        if position == 0:
            # Long setup: RSI crosses above 40 (from below) with 4h uptrend and volume confirmation
            if (rsi_val > 40 and rsi[i-1] <= 40 and  # RSI crossing above 40
                close[i] > ema_4h_val and             # Price above 4h EMA (uptrend)
                vol_1d_current > vol_ma_1d_val):      # Volume above average
                position = 1
                signals[i] = position_size
            # Short setup: RSI crosses below 60 (from above) with 4h downtrend and volume confirmation
            elif (rsi_val < 60 and rsi[i-1] >= 60 and  # RSI crossing below 60
                  close[i] < ema_4h_val and             # Price below 4h EMA (downtrend)
                  vol_1d_current > vol_ma_1d_val):      # Volume above average
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses below 50 or trend changes
            if rsi_val < 50 or close[i] < ema_4h_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses above 50 or trend changes
            if rsi_val > 50 or close[i] > ema_4h_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_4hEMA_1dVolume"
timeframe = "1h"
leverage = 1.0