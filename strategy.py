# 6h_ElderRay_Power_1dTrend_Volume
# Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long: Bull Power > 0, price > 1d EMA50, volume > 1.5x avg
# Short: Bear Power < 0, price < 1d EMA50, volume > 1.5x avg
# Exit when power reverses sign
# Target: 50-150 trades over 4 years (12-37/year)
# Works in bull (trend following) and bear (mean reversion via power reversal)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_Power_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA(13) for Elder Ray (13 periods on 6h timeframe)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Bull Power positive, price above 1d EMA trend, volume confirmation
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, price below 1d EMA trend, volume confirmation
            elif bear_power[i] < 0 and close[i] < ema50_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power becomes negative (trend weakening)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power becomes positive (trend weakening)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals