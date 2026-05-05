#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike (2.0x)
# Long when price breaks above 1h Camarilla R3 AND price > 4h EMA50 (uptrend) AND volume > 2.0x 20-period average
# Short when price breaks below 1h Camarilla S3 AND price < 4h EMA50 (downtrend) AND volume > 2.0x 20-period average
# Exit when price crosses 1h Camarilla pivot point OR EMA50 filter reverses
# Uses Camarilla levels for precise intraday support/resistance, proven effective in ranging and trending markets
# 4h EMA50 provides stronger trend filter than shorter MAs, reducing false signals in chop
# Volume spike threshold of 2.0x ensures momentum confirmation while limiting overtrading
# Session filter (08-20 UTC) avoids low-liquidity periods
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h timeframe
# Timeframe: 1h (primary)
# Target symbols: BTC/ETH/SOL (avoid SOL-only bias)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_2.0x"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data ONCE before loop for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h Camarilla levels (based on previous bar)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close = np.roll(close_1h, 1)
    prev_high[0] = high_1h[0]  # first bar uses current values
    prev_low[0] = low_1h[0]
    prev_close[0] = close_1h[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = camarilla_pivot + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = camarilla_pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    # Get 4h data ONCE before loop for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation on 1h (threshold: 2.0x for stricter filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > EMA50 (uptrend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND price < EMA50 (downtrend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot OR price < EMA50 (trend weakening)
            if close[i] < camarilla_pivot_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot OR price > EMA50 (trend weakening)
            if close[i] > camarilla_pivot_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals