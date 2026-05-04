#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w HMA trend filter and volume confirmation
# Long when price breaks above Camarilla R3 with 1w bullish HMA trend and volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S3 with 1w bearish HMA trend and volume > 1.5x 20-period volume EMA
# Uses 1w HMA(21) for major trend filter to reduce whipsaw, targeting 15-25 trades/year on 1d.
# Volume spike filter (1.5x) is moderate to avoid overtrading while ensuring conviction.
# Camarilla R3/S3 are strong breakout levels from previous day's range.
# Works in bull markets via longs in bullish 1w HMA regime and bear markets via shorts in bearish 1w HMA regime.

name = "1d_Camarilla_R3S3_1wHMA_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF HMA trend - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate HMA(21) on 1w close
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = np.convolve(arr, np.ones(half_period)/half_period, mode='same')
        wma1 = 2 * np.convolve(arr, np.ones(period)/period, mode='same')
        wma3 = np.convolve(2*wma2 - wma1, np.ones(sqrt_period)/sqrt_period, mode='same')
        # Handle edges
        wma2[:half_period-1] = np.nan
        wma2[-half_period+1:] = np.nan
        wma1[:period-1] = np.nan
        wma1[-period+1:] = np.nan
        wma3[:sqrt_period-1] = np.nan
        wma3[-sqrt_period+1:] = np.nan
        return wma3
    hma_21_1w = hma(close_1w, 21)
    
    # Determine 1w HMA trend: bullish if close > HMA, bearish if close < HMA
    hma_bullish_1w = close_1w > hma_21_1w
    hma_bearish_1w = close_1w < hma_21_1w
    
    # Align 1w HMA trend to 1d timeframe
    hma_bullish_aligned = align_htf_to_ltf(prices, df_1w, hma_bullish_1w.astype(float))
    hma_bearish_aligned = align_htf_to_ltf(prices, df_1w, hma_bearish_1w.astype(float))
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 from previous 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1
    camarilla_r3 = close_1d + camarilla_range / 4  # R3 = close + 1.1*(high-low)*1.1/4
    camarilla_s3 = close_1d - camarilla_range / 4  # S3 = close - 1.1*(high-low)*1.1/4
    
    # Align Camarilla levels to 1d timeframe (same timeframe, so just shift by 1 for previous day's values)
    camarilla_r3_aligned = np.roll(camarilla_r3, 1)
    camarilla_s3_aligned = np.roll(camarilla_s3, 1)
    camarilla_r3_aligned[0] = np.nan
    camarilla_s3_aligned[0] = np.nan
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(hma_bullish_aligned[i]) or np.isnan(hma_bearish_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1w bullish HMA AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                hma_bullish_aligned[i] > 0.5 and  # 1w bullish HMA trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1w bearish HMA AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  hma_bearish_aligned[i] > 0.5 and  # 1w bearish HMA trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1w HMA turns bearish
            if (close[i] < camarilla_s3_aligned[i] or 
                hma_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1w HMA turns bullish
            if (close[i] > camarilla_r3_aligned[i] or 
                hma_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals