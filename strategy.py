#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with Daily EMA Filter and Volume Confirmation.
# Uses Williams %R(14) on 6h timeframe for oversold/overbought conditions.
# Filters trades by 6h EMA(50) for trend direction (long above EMA, short below).
# Requires volume > 1.5x 20-period average to ensure quality signals.
# Works in bull/bear markets via EMA filter that adapts to trend direction.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_williamsr_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) calculation
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when no range
    
    # EMA(50) on 6h timeframe
    ema50 = np.full(n, np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(n):
        if i == 0:
            ema50[i] = close[i]
        elif not np.isnan(close[i]):
            ema50[i] = alpha * close[i] + (1 - alpha) * ema50[i-1]
        else:
            ema50[i] = ema50[i-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R crosses above -20 (overbought) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (williams_r[i] >= -20 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses below -80 (oversold) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (williams_r[i] <= -80 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: Williams %R crosses above -80 from below AND price above EMA50
                if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                    close[i] > ema50[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -20 from above AND price below EMA50
                elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                      close[i] < ema50[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals