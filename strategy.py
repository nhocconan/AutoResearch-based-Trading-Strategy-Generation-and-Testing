# 4h_pullback_trend_follow_v1
# Hypothesis: Trend following with pullback entries on 4h timeframe. Long when price pulls back to rising EMA21 during uptrend (price > EMA50). Short when price pulls back to falling EMA21 during downtrend (price < EMA50). Uses EMA21/50 crossovers for trend direction and pulls back to EMA21 for entry. Volume confirmation reduces false signals. Designed to work in both bull (trend following) and bear (shorting downtrends) markets. Target: 25-40 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_pullback_trend_follow_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA21 and EMA50 for trend and pullback
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 1.2x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.2 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(21, 50, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below EMA21 or trend turns down (EMA21 < EMA50)
            if close[i] < ema21[i] or ema21[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above EMA21 or trend turns up (EMA21 > EMA50)
            if close[i] > ema21[i] or ema21[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above EMA50 (uptrend) AND pulls back to EMA21 with volume surge
            if (close[i] > ema50[i] and close[i] <= ema21[i] * 1.005 and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below EMA50 (downtrend) AND pulls back to EMA21 with volume surge
            elif (close[i] < ema50[i] and close[i] >= ema21[i] * 0.995 and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals