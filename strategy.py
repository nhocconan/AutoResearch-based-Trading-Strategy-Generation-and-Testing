#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 with 1w EMA34 uptrend and volume > 2.0x 20-period EMA
# Short when price breaks below Camarilla S3 with 1w EMA34 downtrend and volume > 2.0x 20-period EMA
# Uses 1w HTF for trend to reduce whipsaw vs 1d, targeting 10-25 trades/year on 1d.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.
# Volume confirmation filters false breakouts; ATR-based stoploss via trend reversal.

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Shift by 1 to use previous day's values
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan  # First value has no previous
    
    # Calculate Camarilla levels
    R3 = typical_price_prev + (high - low) * 1.1 / 4.0
    S3 = typical_price_prev - (high - low) * 1.1 / 4.0
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2.0x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(typical_price_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1w uptrend AND volume spike
            if (close[i] > R3[i] and 
                close[i] > ema_34_aligned[i] and  # 1w uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1w downtrend AND volume spike
            elif (close[i] < S3[i] and 
                  close[i] < ema_34_aligned[i] and  # 1w downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1w trend turns down
            if (close[i] < S3[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1w trend turns up
            if (close[i] > R3[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals