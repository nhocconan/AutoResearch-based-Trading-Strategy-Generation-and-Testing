#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Volume + Weekly Trend
# Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# combined with volume confirmation and weekly trend filter provides high-probability
# entries in both trending and ranging markets. The weekly trend ensures we trade
# with the dominant market direction, while Camarilla levels provide precise
# entry/exit points. Volume filters out false breakouts.
# Target: 15-25 trades/year to minimize fee drag on 6h timeframe.
name = "6h_camarilla_pivot_volume_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily OHLC for Camarilla calculation (using 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formula
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_ema_6h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(weekly_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or weekly trend turns bearish
            if close[i] < s3_6h[i] or close[i] < weekly_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or weekly trend turns bullish
            if close[i] > r3_6h[i] or close[i] > weekly_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Mean reversion at S3/R3 in ranging markets
                if close[i] <= s3_6h[i] and close[i] > weekly_ema_6h[i]:
                    # Buy at S3 when above weekly EMA (bullish bias)
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= r3_6h[i] and close[i] < weekly_ema_6h[i]:
                    # Sell at R3 when below weekly EMA (bearish bias)
                    position = -1
                    signals[i] = -0.25
                # Breakout continuation at R4/S4
                elif close[i] > r4_6h[i] and close[i] > weekly_ema_6h[i]:
                    # Buy breakout above R4 in bullish trend
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_6h[i] and close[i] < weekly_ema_6h[i]:
                    # Sell breakdown below S4 in bearish trend
                    position = -1
                    signals[i] = -0.25
    
    return signals