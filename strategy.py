# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Beta
# Hypothesis: Refined version with stricter volume confirmation (3x MA) and reduced position size (0.20) to lower trade frequency and improve generalization.
# Uses 1d Camarilla R3/S3 for breakouts, 1d EMA34 trend filter, and volume > 3x 20-period MA for confirmation.
# Exits when price crosses back over/under 1d EMA. Designed for fewer, higher-quality trades in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Beta"
timeframe = "4h"
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
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d bar OHLC for Camarilla calculation
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].shift(1).values
    
    # Calculate R3 and S3 levels
    rng = phigh - plow
    r3 = pclose + rng * 1.1 / 4.0
    s3 = pclose - rng * 1.1 / 4.0
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    pclose_series = pd.Series(pclose)
    ema1d = pclose_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    # Volume confirmation: 20-period MA on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with trend alignment and strong volume
            if close[i] > r3_aligned[i] and close[i] > ema1d_aligned[i] and volume[i] > vol_ma[i] * 3.0:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S3 with trend alignment and strong volume
            elif close[i] < s3_aligned[i] and close[i] < ema1d_aligned[i] and volume[i] > vol_ma[i] * 3.0:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA (trend invalidated)
            if close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA (trend invalidated)
            if close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals