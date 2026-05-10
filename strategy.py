# 1D_Camarilla_R3_S3_Breakout_1W_EMA50_Trend_VolumeS
# Hypothesis: Uses 1d timeframe with 1w EMA50 trend filter to avoid overtrading and improve robustness.
# Combines Camarilla R3/S3 breakouts on 1d with weekly EMA50 trend filter and volume confirmation.
# Designed for 1d timeframe to target 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals.
# Volume confirmation adds conviction to breakouts. Works in both bull and bear markets by following weekly trend.

name = "1D_Camarilla_R3_S3_Breakout_1W_EMA50_Trend_VolumeS"
timeframe = "1d"
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    camarilla_r3 = close + ((high - low) * 1.1 / 4)
    camarilla_s3 = close - ((high - low) * 1.1 / 4)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + above 1w EMA50 + volume spike
            if (close[i] > camarilla_r3[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below 1w EMA50 + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S3 (re-enters range) or volume drops below average
            if (close[i] < camarilla_s3[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R3 (re-enters range) or volume drops below average
            if (close[i] > camarilla_r3[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals