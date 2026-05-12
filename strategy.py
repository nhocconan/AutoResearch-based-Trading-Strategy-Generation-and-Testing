# 6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike
# Hypothesis: Use 12h Camarilla pivot levels with breakout at R4/S4 levels, filtered by 12h EMA trend and volume spike.
# Long when price breaks above R4 with price > 12h EMA and volume > 2x MA.
# Short when price breaks below S4 with price < 12h EMA and volume > 2x MA.
# Exit when price crosses back to R3/S3 levels (reversion to mean within range).
# Targets 15-30 trades/year to minimize fee drag while capturing strong breakouts in both bull and bear markets.
# Uses 12h timeframe for structure and trend, with 6h for entry timing.

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike"
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
    
    # Calculate 12h Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar's high, low, close
    ph = df_12h['high'].values
    pl = df_12h['low'].values
    pc = df_12h['close'].values
    
    # Camarilla calculations
    range_ = ph - pl
    r4 = pc + range_ * 1.1 / 2
    r3 = pc + range_ * 1.1 / 4
    s3 = pc - range_ * 1.1 / 4
    s4 = pc - range_ * 1.1 / 2
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # 12h EMA for trend filter
    ema12 = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema12_aligned = align_htf_to_ltf(prices, df_12h, ema12)
    
    # Volume confirmation: 20-period moving average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema12_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 with price > 12h EMA and volume > 2x MA
            if close[i] > r4_aligned[i] and close[i] > ema12_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with price < 12h EMA and volume > 2x MA
            elif close[i] < s4_aligned[i] and close[i] < ema12_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below R3 (mean reversion within range)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above S3 (mean reversion within range)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals