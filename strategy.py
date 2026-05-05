#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R3 level AND price > 1w EMA34 AND volume > 1.8 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 level AND price < 1w EMA34 AND volume > 1.8 * avg_volume(20)
# Exit when price crosses 1d Camarilla pivot (midpoint) OR volume < 0.6 * avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year)
# Camarilla levels from 1d provide intraday support/resistance; 1w EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
# Prioritizes BTC/ETH over SOL by requiring HTF trend alignment

name = "1d_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), 
    #            S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # Pivot = (high + low + close) / 3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # handle first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pivot + 1.125 * camarilla_range
    camarilla_s3 = camarilla_pivot - 1.125 * camarilla_range
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    volume_exit = volume < (0.6 * avg_volume_20)  # stricter exit condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after warmup period for EMA34
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1w EMA34, volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1w EMA34, volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla pivot OR volume drops significantly
            if close[i] < camarilla_pivot[i] or volume_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla pivot OR volume drops significantly
            if close[i] > camarilla_pivot[i] or volume_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals