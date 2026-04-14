#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume confirmation
# 4h EMA(21) provides trend bias to avoid counter-trend trades
# 1h RSI(14) with overbought/oversold levels for entry timing
# 1d volume spike (>2x 20-day average) confirms institutional participation
# Works in bull/bear as 4h EMA adapts to trend and volume filters noise
# Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(21) for trend filter
    ema_len = 21
    if len(df_4h) < ema_len:
        return np.zeros(n)
    
    ema_4h = pd.Series(df_4h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume 20-day average for spike detection
    vol_len = 20
    if len(df_1d) < vol_len:
        return np.zeros(n)
    
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=vol_len, min_periods=vol_len).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h RSI(14) for entry timing
    rsi_len = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(50, ema_len, vol_len, rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA21
        above_ema = close[i] > ema_4h_aligned[i]
        below_ema = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation: current 1d volume > 2x average
        volume_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30) + above 4h EMA + volume spike
            if (rsi_values[i] < 30 and 
                above_ema and 
                volume_spike):
                position = 1
                signals[i] = position_size
            # Enter short: RSI overbought (>70) + below 4h EMA + volume spike
            elif (rsi_values[i] > 70 and 
                  below_ema and 
                  volume_spike):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought (>70) or price below 4h EMA
            if rsi_values[i] > 70 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI oversold (<30) or price above 4h EMA
            if rsi_values[i] < 30 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_EMA21_RSI_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0