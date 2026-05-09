#/usr/bin/env python3
# Hypothesis: 1h mean reversion with 4h trend filter and 1d volume regime
# Long when price is below 4h VWAP but above 1h VWAP, volume above 1d average, and 4h trend is bullish
# Short when price is above 4h VWAP but below 1h VWAP, volume above 1d average, and 4h trend is bearish
# Exit when price crosses 1h VWAP or volume drops below average
# Uses 4h for trend direction and 1d for volume regime to reduce whipsaw, 1h for precise entry
# Position size: 0.20 (20% of capital) to limit drawdown
# Designed to work in both trending and ranging markets via VWAP mean reversion and volume confirmation

name = "1h_VWAP_MeanReversion_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

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
    
    # 1h VWAP for mean reversion signal
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap_1h = vwap_num / vwap_den
    
    # 4h VWAP for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_num_4h = np.cumsum(typical_price_4h * df_4h['volume'].values)
    vwap_den_4h = np.cumsum(df_4h['volume'].values)
    vwap_4h = vwap_num_4h / vwap_den_4h
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # 1d volume average for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    vol_avg_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume average
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1h[i]) or np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price below 4h VWAP (bullish 4h context) but above 1h VWAP (short-term mean reversion long)
            # AND volume above 1d average (confirm participation)
            if (close[i] < vwap_4h_aligned[i] and 
                close[i] > vwap_1h[i] and 
                volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price above 4h VWAP (bearish 4h context) but below 1h VWAP (short-term mean reversion short)
            # AND volume above 1d average (confirm participation)
            elif (close[i] > vwap_4h_aligned[i] and 
                  close[i] < vwap_1h[i] and 
                  volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1h VWAP (mean reversion complete) OR volume drops below average
            if (close[i] < vwap_1h[i]) or (volume[i] < vol_avg_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 1h VWAP (mean reversion complete) OR volume drops below average
            if (close[i] > vwap_1h[i]) or (volume[i] < vol_avg_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals