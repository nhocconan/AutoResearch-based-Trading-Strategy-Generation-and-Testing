#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Wilder's Parabolic SAR for trend detection with volume confirmation.
- Parabolic SAR identifies trend direction and provides trailing stops
- Enter long when price is above SAR, volume > 1.5x 20-period volume MA, and closing price > opening price (bullish candle)
- Enter short when price is below SAR, volume > 1.5x 20-period volume MA, and closing price < opening price (bearish candle)
- Exit when price crosses SAR (trend reversal)
- Fixed position size 0.25 to manage drawdown
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
"""

import numpy as np
import pandas as pd
from math import exp, log

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize arrays
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    ep = np.zeros(n)     # extreme point
    af = np.zeros(n)     # acceleration factor
    
    # Initialize first values
    trend[0] = 1 if close[1] > close[0] else -1
    sar[0] = low[0] if trend[0] == 1 else high[0]
    ep[0] = high[0] if trend[0] == 1 else low[0]
    af[0] = af_start
    
    # Calculate Parabolic SAR
    for i in range(1, n):
        # SAR calculation
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        
        # Determine current trend
        if trend[i-1] == 1:  # was uptrend
            if low[i] <= sar[i]:  # trend reversal to downtrend
                trend[i] = -1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]
                af[i] = af_start
            else:  # continue uptrend
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # was downtrend
            if high[i] >= sar[i]:  # trend reversal to uptrend
                trend[i] = 1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]
                af[i] = af_start
            else:  # continue downtrend
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        
        # Ensure SAR doesn't penetrate the last two periods' low/high
        if trend[i] == 1:  # uptrend
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:  # downtrend
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
    
    # Volume confirmation: 20-period volume MA
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(volume_ma_20[i]) or np.isnan(sar[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        sar_val = sar[i]
        is_bullish_candle = close[i] > open_price[i]
        is_bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Look for entries with volume confirmation and candle direction
            # Long: price above SAR + volume spike + bullish candle
            if price > sar_val and vol > 1.5 * vol_ma and is_bullish_candle:
                signals[i] = 0.25
                position = 1
            # Short: price below SAR + volume spike + bearish candle
            elif price < sar_val and vol > 1.5 * vol_ma and is_bearish_candle:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below SAR (trend reversal)
            if price < sar_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above SAR (trend reversal)
            if price > sar_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ParabolicSAR_Volume_Candle"
timeframe = "4h"
leverage = 1.0