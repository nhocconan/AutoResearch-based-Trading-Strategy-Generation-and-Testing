#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 Trend Filter and Volume Spike
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA34 AND volume spike
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA34 AND volume spike
# Williams Alligator uses smoothed medians (SMMA) of 13, 8, 5 periods to identify trend phases
# 1d EMA34 provides higher timeframe trend filter to reduce whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends
# Works in bull (trend + alignment) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 12h (primary timeframe as required)

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: SMMA of median price (hlc3)
    hlc3_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    
    # Smoothed Moving Average (SMMA) - also called Wilder's MA or RMA
    def smma(source, period):
        n = len(source)
        result = np.full(n, np.nan)
        if n < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, n):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Alligator lines: Jaw (13, 8), Teeth (8, 5), Lips (5, 3)
    jaw = smma(hlc3_12h, 13)  # Blue line
    teeth = smma(hlc3_12h, 8)  # Red line
    lips = smma(hlc3_12h, 5)   # Green line
    
    # Align Alligator lines to 12h timeframe (already on 12h, so direct use)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for SMMA)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips
            bullish = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
            # Bearish alignment: Jaw > Teeth > Lips
            bearish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
            
            # Long: bullish alignment AND price > 1d EMA34 AND volume spike
            if (bullish and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < 1d EMA34 AND volume spike
            elif (bearish and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment turns bearish OR price closes below 1d EMA34
            bearish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
            if bearish or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment turns bullish OR price closes above 1d EMA34
            bullish = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
            if bullish or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals