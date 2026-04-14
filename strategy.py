#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume confirmation
# Uses 1h RSI for entry timing, 4h EMA for trend direction, 1d volume spike for confirmation
# Only trades during 08-20 UTC to avoid low-liquidity hours
# Designed to work in both bull (trend following) and bear (mean reversion in ranges)
# Target: 15-35 trades per year (60-140 over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend (21-period)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long: RSI oversold in uptrend OR RSI overbought in downtrend (mean reversion)
            if (ema_4h_aligned[i] > ema_4h_aligned[i-1] and  # Uptrend
                rsi[i] < 30 and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):  # Volume spike
                position = 1
                signals[i] = position_size
            # Short: RSI overbought in downtrend OR RSI oversold in uptrend
            elif (ema_4h_aligned[i] < ema_4h_aligned[i-1] and  # Downtrend
                  rsi[i] > 70 and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):  # Volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi[i] > 70 or ema_4h_aligned[i] < ema_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi[i] < 30 or ema_4h_aligned[i] > ema_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_4hEMA_1dVolume"
timeframe = "1h"
leverage = 1.0