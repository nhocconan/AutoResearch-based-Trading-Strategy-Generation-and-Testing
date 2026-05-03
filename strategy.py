#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume confirmation
# Uses 4h primary timeframe with 12h HTF for trend alignment. Camarilla pivot levels provide
# institutional support/resistance. Breakouts above R3 or below S3 in direction of 12h EMA(50)
# with volume confirmation (2.0x 20-period average) capture strong moves. Designed for low trade
# frequency (19-50/year) to minimize fee drag. Works in both bull and bear markets by following
# the 12h trend direction only.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 51:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels on 4h using previous bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as breakout levels
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    camarilla_high = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_low = prev_close - 1.1 * (prev_high - prev_low)
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 51  # max(50 for EMA, 1 for shift in Camarilla, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + above 12h EMA(50) + volume spike
            if (close[i] > camarilla_high[i] and close[i] > ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 + below 12h EMA(50) + volume spike
            elif (close[i] < camarilla_low[i] and close[i] < ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Camarilla S3 (mean reversion) or below 12h EMA(50) (trend reversal)
            if close[i] < camarilla_low[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Camarilla R3 (mean reversion) or above 12h EMA(50) (trend reversal)
            if close[i] > camarilla_high[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals