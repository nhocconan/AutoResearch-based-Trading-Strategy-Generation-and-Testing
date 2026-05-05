#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) crossover + 1w EMA34 trend filter + volume confirmation
# Long when Alligator Lips cross above Teeth AND price > Jaw AND 1w close > 1w EMA34 AND volume > 2x 20-period average
# Short when Alligator Lips cross below Teeth AND price < Jaw AND 1w close < 1w EMA34 AND volume > 2x 20-period average
# Exit when Alligator Lips cross back over Teeth (reversal signal)
# Uses 1d primary timeframe with 1w HTF for trend filter
# Williams Alligator catches trends early with smoothed moving averages, reducing whipsaw
# Volume confirmation filters false breakouts, trend filter ensures alignment with weekly direction
# Discrete sizing (0.25) to limit fee drag and manage drawdown in bear markets
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_WilliamsAlligator_JawTeeth_Lips_Cross_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator on 1d data: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips cross above Teeth AND price > Jaw AND 1w close > 1w EMA34 AND volume spike
            if (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and  # crossover up
                close[i] > jaw[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips cross below Teeth AND price < Jaw AND 1w close < 1w EMA34 AND volume spike
            elif (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and  # crossover down
                  close[i] < jaw[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips cross back below Teeth (reversal signal)
            if lips[i] < teeth[i] and lips[i-1] >= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips cross back above Teeth (reversal signal)
            if lips[i] > teeth[i] and lips[i-1] <= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals