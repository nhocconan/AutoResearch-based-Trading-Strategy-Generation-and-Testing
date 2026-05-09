#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_Filter_Volume_Confirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h KAMA trend filter with volume confirmation and 1d ATR filter.
    - Long: KAMA > previous KAMA, volume > 1.5x avg, close > 1d EMA(50)
    - Short: KAMA < previous KAMA, volume > 1.5x avg, close < 1d EMA(50)
    - Exit: Trend reversal or volume drops below average
    - Uses 1d EMA(50) for trend filter
    - Target: 20-40 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate KAMA on 12h data
    close_series = pd.Series(close)
    # Efficiency ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising, volume confirmation, above 1d EMA trend
            if kama_up and vol_ok and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, volume confirmation, below 1d EMA trend
            elif kama_down and vol_ok and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down or volume drops
            if not kama_up or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up or volume drops
            if not kama_down or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals