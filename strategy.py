#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance; breakouts with volume confirm institutional interest
# 1d EMA34 ensures we trade with higher timeframe trend to avoid whipsaws in ranging markets
# Volume spike (>1.8x 20-period EMA) filters false breakouts
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# Works in bull/bear: trend filter prevents counter-trend entries during strong moves

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(open_price, 1)
    # First bar: use current values (will be overwritten anyway as we start from 50)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = open_price[0]
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    r4 = prev_close + 1.1 * camarilla_range * 1.1 / 2
    s4 = prev_close - 1.1 * camarilla_range * 1.1 / 2
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume spike + above 1d EMA34
            if close[i] > r3[i] and volume_spike and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume spike + below 1d EMA34
            elif close[i] < s3[i] and volume_spike and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) OR below 1d EMA34 (trend change)
            if close[i] < s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) OR above 1d EMA34 (trend change)
            if close[i] > r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals