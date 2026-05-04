#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with 4h EMA50 uptrend and volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S3 level with 4h EMA50 downtrend and volume > 1.5x 20-period volume EMA
# Uses 4h HTF for trend to reduce whipsaw vs 1d, targeting 15-37 trades/year on 1h.
# Volume spike filter (1.5x) is moderate to avoid overtrading. Camarilla levels provide institutional support/resistance.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "1h_Camarilla_R3S3_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels for 1h using previous bar's OHLC
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # avoid NaN on first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade during session hours
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 4h uptrend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_50_aligned[i] and  # 4h uptrend
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 4h downtrend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_50_aligned[i] and  # 4h downtrend
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 4h trend turns down
            if (close[i] < camarilla_s3[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 4h trend turns up
            if (close[i] > camarilla_r3[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals