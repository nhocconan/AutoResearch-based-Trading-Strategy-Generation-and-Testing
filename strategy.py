#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Keltner Channels with ATR-based volatility filter
# - Long when price breaks above upper Keltner Channel with ATR(14) > 1.0 and volume spike
# - Short when price breaks below lower Keltner Channel with ATR(14) > 1.0 and volume spike
# - Exit when price crosses back inside Keltner Channels
# - Designed to capture volatility breakouts in both trending and ranging markets
# - Uses 1d Keltner Channels (EMA20 + 2*ATR(10)) for robust volatility-based bands
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_KeltnerBreakout_ATR14_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for Keltner Channel center
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR(10) for Keltner Channel width
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d Keltner Channels
    kc_upper_1d = ema_20_1d + (2 * atr_10_1d)
    kc_lower_1d = ema_20_1d - (2 * atr_10_1d)
    
    # Align 1d indicators to 4h timeframe
    kc_upper_4h = align_htf_to_ltf(prices, df_1d, kc_upper_1d)
    kc_lower_4h = align_htf_to_ltf(prices, df_1d, kc_lower_1d)
    
    # Calculate 4h ATR(14) for volatility filter (must be > 1.0)
    tr1_4h = np.abs(high[1:] - low[1:])
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(kc_upper_4h[i]) or np.isnan(kc_lower_4h[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner Channel with ATR filter and volume spike
            if close[i] > kc_upper_4h[i] and atr_14_4h[i] > 1.0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner Channel with ATR filter and volume spike
            elif close[i] < kc_lower_4h[i] and atr_14_4h[i] > 1.0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back inside Keltner Channel (below upper)
            if close[i] < kc_upper_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back inside Keltner Channel (above lower)
            if close[i] > kc_lower_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals