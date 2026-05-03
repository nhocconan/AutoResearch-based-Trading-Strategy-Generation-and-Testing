#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Williams Alligator identifies trend presence via smoothed medians (Jaw/Teeth/Lips)
# When all three lines are aligned (bullish: Lips > Teeth > Jaw, bearish: reverse), trend is strong
# 1d EMA(34) ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability entries
# Works in bull/bear markets by following 1d trend direction for entries
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator from 1d data
    # Alligator uses SMMA (Smoothed Moving Average) with specific periods
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = smma(median_price_1d, 13)  # Jaw: SMMA(13, 8)
    teeth = smma(median_price_1d, 8)  # Teeth: SMMA(8, 5)
    lips = smma(median_price_1d, 5)   # Lips: SMMA(5, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips, additional_delay_bars=0)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Enter long: Bullish alignment + price above 1d EMA34 + volume spike
            if bullish_aligned and close[i] > ema_34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish alignment + price below 1d EMA34 + volume spike
            elif bearish_aligned and close[i] < ema_34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price below 1d EMA34
            if bearish_aligned or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR price above 1d EMA34
            if bullish_aligned or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals