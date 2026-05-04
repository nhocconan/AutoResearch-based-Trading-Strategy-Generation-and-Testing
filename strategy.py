#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with 1d bullish trend (close > EMA34) and volume > 1.8x 24-period volume EMA
# Short when price breaks below Camarilla S3 level with 1d bearish trend (close < EMA34) and volume > 1.8x 24-period volume EMA
# Uses 1d EMA34 for trend filter to reduce whipsaw, targeting 12-37 trades/year on 12h.
# Volume spike filter (1.8x) is strict to avoid overtrading. Camarilla levels provide intraday structure.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.

name = "12h_Camarilla_R3S3_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1d = close_1d > ema_34_1d
    trend_bearish_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Camarilla levels from previous 12h bar (using typical price)
    # Camarilla formula: Pivot = (H+L+C)/3
    # R3 = Pivot + (H-L)*1.1/2
    # S3 = Pivot - (H-L)*1.1/2
    typical_price = (high + low + close) / 3
    pivot = pd.Series(typical_price).shift(1).rolling(window=1, min_periods=1).mean().values  # previous bar
    range_hl = (high - low).shift(1)  # previous bar range
    camarilla_r3 = pivot + (range_hl * 1.1 / 2)
    camarilla_s3 = pivot - (range_hl * 1.1 / 2)
    
    # Calculate volume spike filter (24-period volume EMA = 2 periods of 12h)
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (vol_ema_24 * 1.8)  # Volume at least 1.8x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d bullish trend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d bearish trend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1d trend turns bearish
            if (close[i] < camarilla_s3[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1d trend turns bullish
            if (close[i] > camarilla_r3[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals