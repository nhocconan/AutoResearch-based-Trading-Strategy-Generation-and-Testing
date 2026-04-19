#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND price > 1d EMA50 (uptrend) AND volume > 1.3x 1d average volume.
# Short when Bear Power < 0 (close < EMA13) AND price < 1d EMA50 (downtrend) AND volume > 1.3x 1d average volume.
# Exit when Bull/Bear Power crosses zero (close crosses EMA13).
# Uses Elder Ray for momentum, 1d EMA for trend filter, volume for confirmation.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).

name = "6h_ElderRay_EMA13_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA13 for Elder Ray calculation
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = close - ema13
    bear_power = ema13 - close
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 1d average volume for confirmation (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        if position == 0:
            # Long entry: Bull Power positive + price above 1d EMA50 + volume spike
            if bp > 0 and price > ema50_val and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power positive + price below 1d EMA50 + volume spike
            elif br > 0 and price < ema50_val and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below zero (close crosses below EMA13)
            if bp <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power crosses below zero (close crosses above EMA13)
            if br <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals