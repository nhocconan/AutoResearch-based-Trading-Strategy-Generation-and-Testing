#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_With_Volume_and_1wTrend_v1
Hypothesis: Buy when price breaks above upper Keltner Channel (20,2) with volume spike and above 1w EMA40 trend; sell when price breaks below lower channel with volume spike and below 1w EMA40. Keltner Channels adapt to volatility better than Bollinger Bands in trending markets, volume confirms institutional participation, and 1w EMA40 ensures alignment with long-term trend. Designed for low trade frequency (<15/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2)
    close_series = pd.Series(close)
    ma = close_series.rolling(window=20, min_periods=20).mean()
    atr_series = pd.Series(high - low).rolling(window=20, min_periods=20).mean()
    upper = ma + 2 * atr_series
    lower = ma - 2 * atr_series
    upper_kc = upper.values
    lower_kc = lower.values
    middle_kc = ma.values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1w EMA40 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need KC and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(upper_kc[i]) or
            np.isnan(lower_kc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        upper = upper_kc[i]
        lower = lower_kc[i]
        middle = middle_kc[i]
        
        if position == 0:
            # Long: price > upper KC with volume spike and above 1w EMA40
            if price > upper and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price < lower KC with volume spike and below 1w EMA40
            elif price < lower and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < middle KC or below 1w EMA40
            if price < middle or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > middle KC or above 1w EMA40
            if price > middle or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Keltner_Channel_Breakout_With_Volume_and_1wTrend_v1"
timeframe = "1d"
leverage = 1.0