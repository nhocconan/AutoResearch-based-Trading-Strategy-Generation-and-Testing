#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 1d EMA34 > EMA55 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 1d EMA34 < EMA55 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1d EMA34 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Camarilla provides strong intraday support/resistance levels with clear breakout zones
# 1d EMA34/EMA55 filter ensures alignment with intermediate daily trend
# Volume confirmation filters weak breakouts
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend)

name = "12h_1dCamarillaR3S3_1dEMA34Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA55
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Pivot point = (high + low + close) / 3
    # R3 = high + 2*(pivot - low)
    # S3 = low - 2*(high - pivot)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pivot_1d = typical_price_1d
    camarilla_r3_1d = high_1d + 2.0 * (pivot_1d - low_1d)
    camarilla_s3_1d = low_1d - 2.0 * (high_1d - pivot_1d)
    
    # Calculate 1d EMA34 and EMA55 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55_1d = close_series_1d.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_55_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 1d EMA34 > EMA55 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 1d EMA34 < EMA55 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals