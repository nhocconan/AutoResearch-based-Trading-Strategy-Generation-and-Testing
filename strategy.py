#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND close > 1d EMA34 (uptrend) AND volume > 1.5 * 20-bar avg volume
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND close < 1d EMA34 (downtrend) AND volume > 1.5 * 20-bar avg volume
# Exit when Alligator alignment breaks (jaws > teeth or teeth < lips) or price crosses lips in opposite direction
# Williams Alligator identifies trend phases with smoothed median prices, reducing whipsaw in ranging markets
# 1d EMA34 provides higher-timeframe trend filter for better regime adaptation
# Volume confirmation threshold set to 1.5x to balance signal quality and trade frequency
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (smoothed medians)
    # Jaw (blue): 13-period SMMA of median price, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA of median price, shifted 5 bars ahead  
    # Lips (green): 5-period SMMA of median price, shifted 3 bars ahead
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator specification (jaw: 8, teeth: 5, lips: 3)
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full(len(jaw), np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full(len(teeth), np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full(len(lips), np.nan)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams Alligator signals with trend and volume filters
            # Long: Bullish alignment (jaw < teeth < lips) AND price > lips AND uptrend AND volume spike
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and close[i] > lips[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (jaw > teeth > lips) AND price < lips AND downtrend AND volume spike
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < lips[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks bearish OR price crosses below lips
            if jaw[i] > teeth[i] or teeth[i] > lips[i] or close[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks bullish OR price crosses above lips
            if jaw[i] < teeth[i] or teeth[i] < lips[i] or close[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals