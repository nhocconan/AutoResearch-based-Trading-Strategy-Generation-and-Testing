#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume spike (2.0x)
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume spike
# Camarilla levels provide institutional support/resistance; 1d EMA34 filters higher timeframe trend
# Volume spike confirms participation at breakout points
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 12h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (using typical price)
    typical_price = (high + low + close) / 3.0
    if len(typical_price) >= 2:
        # Previous day's typical price
        prev_typical = np.roll(typical_price, 1)
        prev_typical[0] = np.nan  # First value invalid
        # Calculate range
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        # Camarilla levels
        camarilla_base = prev_typical
        camarilla_range = prev_high - prev_low
        r3 = camarilla_base + (camarilla_range * 1.1 / 4)
        s3 = camarilla_base - (camarilla_range * 1.1 / 4)
    else:
        r3 = np.full(n, np.nan)
        s3 = np.full(n, np.nan)
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR price < 1d EMA34 (trend break)
            if close[i] < s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR price > 1d EMA34 (trend break)
            if close[i] > r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals