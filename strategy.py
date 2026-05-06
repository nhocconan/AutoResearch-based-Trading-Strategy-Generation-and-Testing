#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R3 AND 1d close > 1d EMA34 AND volume > 2.0 * 24-bar average volume
# Short when price breaks below S3 AND 1d close < 1d EMA34 AND volume > 2.0 * 24-bar average volume
# Exit when price reverts to 1d EMA34 level or opposite Camarilla level is touched
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide precise intraday support/resistance derived from prior day range
# 1d EMA34 filters for higher timeframe trend alignment to reduce whipsaws
# Volume spike confirmation ensures institutional participation and reduces false breakouts
# Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior 12h bar (using prior bar's range)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C, H, L are from the prior completed 12h bar
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = close[0]  # initialize first bar
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    
    camarilla_range = (prior_high - prior_low) * 1.1 / 2
    r3 = prior_close + camarilla_range  # Resistance level 3
    s3 = prior_close - camarilla_range  # Support level 3
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 24-bar average volume (24 * 12h = 12d approx)
    volume_series = pd.Series(volume)
    avg_volume_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirmation = volume > (2.0 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with trend and volume confirmation
            if close[i] > r3[i] and close[i] > ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with trend and volume confirmation
            elif close[i] < s3[i] and close[i] < ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to 1d EMA34 or touches S3 (mean reversion)
            if close[i] <= ema34_1d_aligned[i] or close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to 1d EMA34 or touches R3 (mean reversion)
            if close[i] >= ema34_1d_aligned[i] or close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals