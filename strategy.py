#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR filter
# Captures breakouts in trending markets, volume confirms institutional interest,
# ATR filter avoids false breakouts in low volatility. Target: 20-40 trades/year.
name = "4h_1d_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume spike: volume > 2.0x 20-day average (moderate to balance trades)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'] > (vol_ma_1d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 4h Donchian channels (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    # 4h ATR for volatility filter (14-period)
    atr_len = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[atr_len-1] = np.mean(tr[:atr_len])
    for i in range(atr_len, n):
        atr[i] = (atr[i-1] * (atr_len - 1) + tr[i]) / atr_len
    
    # ATR filter: only trade when volatility is above average
    atr_ma = np.zeros(n)
    atr_ma[atr_len-1] = np.mean(atr[:atr_len])
    for i in range(atr_len, n):
        atr_ma[i] = (atr_ma[i-1] * (atr_len - 1) + atr[i]) / atr_len
    atr_filter = atr > (atr_ma * 0.8)  # trade when ATR > 80% of its MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(donchian_len, atr_len), n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above upper Donchian with volume spike and sufficient volatility
        long_signal = close[i] > upper[i] and vol_spike_aligned[i] and atr_filter[i]
        # Short: break below lower Donchian with volume spike and sufficient volatility
        short_signal = close[i] < lower[i] and vol_spike_aligned[i] and atr_filter[i]
        
        # Exit when price returns to middle of Donchian channel
        middle = (upper[i] + lower[i]) / 2.0
        exit_long = close[i] < middle
        exit_short = close[i] > middle
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals