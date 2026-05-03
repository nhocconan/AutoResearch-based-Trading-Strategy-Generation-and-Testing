#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend + volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trend via smoothed MAs
# Jaw=13, Teeth=8, Lips=5 periods SMMA (smoothed moving average)
# In uptrend: Lips > Teeth > Jaw; Downtrend: Lips < Teeth < Jaw
# 1d EMA34 ensures alignment with higher timeframe trend
# Volume spike (>1.8x 20-period EMA) confirms breakout strength
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag
# Works in bull/bear: Alligator catches trends, volume filters false breakouts,
# HTF EMA prevents counter-trend trading

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    # SMMA today = (SMMA yesterday * (period-1) + close today) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw(13), Teeth(8), Lips(5)
    # Using median price (typical price) as input
    typical_price = (high + low + close) / 3.0
    jaw = smma(typical_price, 13)  # Blue line
    teeth = smma(typical_price, 8)  # Red line
    lips = smma(typical_price, 5)   # Green line
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid Alligator values
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Alligator trend signals with 1d trend filter
        # Long: Lips > Teeth > Jaw (uptrend) + price above 1d EMA34 + volume spike
        # Short: Lips < Teeth < Jaw (downtrend) + price below 1d EMA34 + volume spike
        if position == 0:
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator reverses (Lips < Teeth) OR price below 1d EMA34
            if lips[i] < teeth[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator reverses (Lips > Teeth) OR price above 1d EMA34
            if lips[i] > teeth[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals