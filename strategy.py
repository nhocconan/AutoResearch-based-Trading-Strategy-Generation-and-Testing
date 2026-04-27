# 12h_WeeklyTrend_VolumeVolatilityFilter_v3
# Hypothesis: 12h timeframe with weekly trend filter, volume confirmation, and volatility filter
# Weekly trend (1w EMA34) determines direction, volume surge confirms momentum, 
# volatility filter avoids choppy markets. Designed for fewer trades (<200 total) to avoid fee drag.
# Works in bull (trend follows) and bear (avoids false signals in chop).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for volume and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily average volume for volume spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-day average (volume surge)
        vol_filter = volume[i] > (vol_ma * 1.5)
        
        # Volatility filter: ATR > 20-period median (avoid extremely low volatility)
        if i >= 20:
            atr_ma = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_val
        vol_regime_filter = atr_val > (atr_ma * 0.5)  # Avoid only extremely quiet periods
        
        # Entry conditions
        if position == 0:
            # Long: weekly uptrend + volume surge + reasonable volatility
            if close[i] > ema_trend and vol_filter and vol_regime_filter:
                signals[i] = size
                position = 1
            # Short: weekly downtrend + volume surge + reasonable volatility
            elif close[i] < ema_trend and vol_filter and vol_regime_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reversal or volume collapse
            if close[i] < ema_trend or volume[i] < (vol_ma * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend reversal or volume collapse
            if close[i] > ema_trend or volume[i] < (vol_ma * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WeeklyTrend_VolumeVolatilityFilter_v3"
timeframe = "12h"
leverage = 1.0