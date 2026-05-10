# 6H_DonchianBreakout_12hTrend_Confirmation
# Hypothesis: Use 6h Donchian(20) breakout confirmed by 12h EMA(50) trend direction and volume spike >2x average.
# Designed for 6h timeframe to capture trend continuation moves with low trade frequency (target: 25-50 trades/year).
# Works in both bull and bear markets by following 12h trend direction, avoiding counter-trend trades.
# Uses discrete position sizing (0.25) to minimize fee churn.

name = "6H_DonchianBreakout_12hTrend_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 6h chart
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 20-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 12h EMA + volume spike
            if (close[i] > donchian_high[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 12h EMA + volume spike
            elif (close[i] < donchian_low[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or volume drops below average
            if (close[i] < donchian_low[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or volume drops below average
            if (close[i] > donchian_high[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals