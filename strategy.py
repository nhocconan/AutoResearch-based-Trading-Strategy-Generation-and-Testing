# 6H_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_v2
# Hypothesis: Combines 12h EMA50 trend filter with Camarilla R3/S3 breakouts and volume confirmation on 6h.
# Uses higher timeframe trend (12h EMA50) to filter direction, reducing false signals in choppy markets.
# Targets 12-37 trades per year on 6h timeframe with discrete position sizing (0.25) to minimize fee churn.
# Designed to work in both bull and bear markets by aligning with 12h trend and requiring volume confirmation.

name = "6H_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_v2"
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
    
    # Get 12h data for EMA trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    camarilla_r3 = df_12h['close'] + ((df_12h['high'] - df_12h['low']) * 1.1 / 4)
    camarilla_s3 = df_12h['close'] - ((df_12h['high'] - df_12h['low']) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (use prior 12h bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    
    # Volume filter: volume > 2.0x 50-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + above 12h EMA50 + volume spike
            if (close[i] > r3_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below 12h EMA50 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S3 (re-enters range) or volume drops below average
            if (close[i] < s3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R3 (re-enters range) or volume drops below average
            if (close[i] > r3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals