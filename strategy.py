#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and 1d volume confirmation (>1.5x 20-period average).
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3). 
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA50 (bullish trend) AND volume > 1.5x MA20.
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA50 (bearish trend) AND volume > 1.5x MA20.
# Exit when Alligator alignment breaks (Lips crosses Teeth) OR price crosses 1w EMA50 in opposite direction.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Williams Alligator identifies trend phases via smoothed SMAs, effective in trending and ranging markets when combined with HTF trend filter.

name = "1d_WilliamsAlligator_1wEMA50_1dVolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Williams Alligator components
    # Jaw: SMA(13,8) - slowest
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8,5) - medium
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5,3) - fastest
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1d volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) - trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_confirm_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish alignment (Lips > Teeth > Jaw) AND close > 1w EMA50 (bullish trend) AND volume confirm
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment (Lips < Teeth < Jaw) AND close < 1w EMA50 (bearish trend) AND volume confirm
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bullish alignment breaks (Lips <= Teeth) OR price < 1w EMA50 (trend change)
            if (lips[i] <= teeth[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bearish alignment breaks (Lips >= Teeth) OR price > 1w EMA50 (trend change)
            if (lips[i] >= teeth[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals