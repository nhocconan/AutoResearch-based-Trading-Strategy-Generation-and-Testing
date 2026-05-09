#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; EMA50 on 1d confirms trend direction.
# Volume spikes (>1.8x average) confirm institutional interest. Designed for low trade frequency.
name = "4h_WilliamsR_1dEMA50_VolumeConfirm"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    # For Williams %R, we need the highest high and lowest low over the lookback period
    # Using rolling window approach
    highest_high_series = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low_series = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll use the inverse for easier interpretation: values below -20 = oversold, above -80 = overbought
    williams_r = (highest_high_series - close) / (highest_high_series - lowest_low_series) * -100
    # Handle division by zero
    williams_r = np.where((highest_high_series - lowest_low_series) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period)  # Need 50 for EMA50 and 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_50_1d_aligned[i]
        wr = williams_r[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.8x average
            if wr < -80 and close[i] > ema_1d and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.8x average
            elif wr > -20 and close[i] < ema_1d and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) OR trend reverses (price < 1d EMA50)
            if wr > -20 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) OR trend reverses (price > 1d EMA50)
            if wr < -80 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals