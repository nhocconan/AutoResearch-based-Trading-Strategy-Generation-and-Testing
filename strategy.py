#!/usr/bin/env python3
# 4h_cci_mean_reversion_v1
# Hypothesis: 4h CCI mean reversion with volume confirmation and ATR stoploss works in both bull and bear markets.
# Long: CCI(20) < -100 + price > EMA(50) + volume > 1.5x 20-period average
# Short: CCI(20) > 100 + price < EMA(50) + volume > 1.5x 20-period average
# Exit: CCI crosses back above -50 (long) or below 50 (short) OR ATR stoploss (2.0x ATR)
# Uses 4h primary timeframe. Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd

name = "4h_cci_mean_reversion_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate EMA(50) for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate CCI(20)
    typical_price = (high + low + close) / 3.0
    tp_s = pd.Series(typical_price)
    sma20 = tp_s.rolling(window=20, min_periods=20).mean().values
    mad = tp_s.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = np.full(n, np.nan)
    for i in range(20, n):
        if mad[i] != 0:
            cci[i] = (typical_price[i] - sma20[i]) / (0.015 * mad[i])
        else:
            cci[i] = 0.0
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        cci_val = cci[i]
        ema50_val = ema50[i]
        atr_val = atr[i]
        
        if np.isnan(vol_r) or np.isnan(cci_val) or np.isnan(ema50_val) or np.isnan(atr_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses back above -50 OR stoploss hit (2.0x ATR below entry)
            if cci_val > -50 or price <= entry_price - 2.0 * atr_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses back below 50 OR stoploss hit (2.0x ATR above entry)
            if cci_val < 50 or price >= entry_price + 2.0 * atr_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: CCI < -100 + price > EMA(50) + volume spike
            if cci_val < -100 and price > ema50_val and vol_r > 1.5:
                position = 1
                entry_price = price
                atr_stop = atr_val
                signals[i] = 0.25
            # Short entry: CCI > 100 + price < EMA(50) + volume spike
            elif cci_val > 100 and price < ema50_val and vol_r > 1.5:
                position = -1
                entry_price = price
                atr_stop = atr_val
                signals[i] = -0.25
    
    return signals