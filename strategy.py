# 4h_4h_WilLIAMS_R_34_Combined
# Hypothesis: Combines Williams %R(34) on 4h with price relative to 4h SMA(50) and volume confirmation.
# Williams %R identifies overbought/oversold conditions; SMA(50) provides trend filter.
# Long when %R < -80 (oversold) and price > SMA(50); Short when %R > -20 (overbought) and price < SMA(50).
# Volume > 1.5x average confirms momentum. Designed for 4h timeframe to avoid overtrading.
# Works in bull markets via trend-following longs; in bear via mean-reversion shorts at resistance.
# Target: 20-40 trades/year per symbol with disciplined risk management.

name = "4h_4h_WilLIAMS_R_34_Combined"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(34) on 4h
    period = 34
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(period-1, len(high)):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    williams_r = np.full_like(close, np.nan)
    denominator = highest_high - lowest_low
    valid = (~np.isnan(denominator)) & (denominator != 0)
    williams_r[valid] = -100 * (highest_high[valid] - close[valid]) / denominator[valid]
    
    # Calculate SMA(50) on 4h
    sma_period = 50
    sma = np.full_like(close, np.nan)
    if len(close) >= sma_period:
        sma[sma_period-1] = np.mean(close[0:sma_period])
        for i in range(sma_period, len(close)):
            sma[i] = (sma[i-1] * (sma_period-1) + close[i]) / sma_period
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # Need SMA, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(sma[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r[i]
        price_above_sma = close[i] > sma[i]
        price_below_sma = close[i] < sma[i]
        vol_confirmed = volume_ratio[i] > 1.5
        
        if position == 0:
            # Enter long: oversold + price above SMA + volume confirmation
            if williams_r_val < -80 and price_above_sma and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + price below SMA + volume confirmation
            elif williams_r_val > -20 and price_below_sma and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R recovers above -50 or price crosses below SMA
            if williams_r_val > -50 or close[i] < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R falls below -50 or price crosses above SMA
            if williams_r_val < -50 or close[i] > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals