#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w trend filter (HMA21) and volume spike confirmation
# Long when: price breaks above Camarilla R1 (1d) AND price > 1w HMA21 (uptrend) AND volume > 2x 20-period MA
# Short when: price breaks below Camarilla S1 (1d) AND price < 1w HMA21 (downtrend) AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla Pivot Point (1d) OR trend reverses
# Uses Camarilla levels for intraday structure, 1w HMA for trend filter, volume spike for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Camarilla_R1S1_Breakout_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels on 1d (using previous day's OHLC)
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Previous day's OHLC
        prev_close = np.roll(close, 1)
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close[0] = np.nan
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        
        # Camarilla calculations
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Resistance levels
        R1 = pivot + (range_hl * 1.0 / 12.0)
        R2 = pivot + (range_hl * 2.0 / 12.0)
        R3 = pivot + (range_hl * 3.0 / 12.0)
        R4 = pivot + (range_hl * 4.0 / 12.0)
        
        # Support levels
        S1 = pivot - (range_hl * 1.0 / 12.0)
        S2 = pivot - (range_hl * 2.0 / 12.0)
        S3 = pivot - (range_hl * 3.0 / 12.0)
        S4 = pivot - (range_hl * 4.0 / 12.0)
    else:
        pivot = np.full(n, np.nan)
        R1 = np.full(n, np.nan)
        R2 = np.full(n, np.nan)
        R3 = np.full(n, np.nan)
        R4 = np.full(n, np.nan)
        S1 = np.full(n, np.nan)
        S2 = np.full(n, np.nan)
        S3 = np.full(n, np.nan)
        S4 = np.full(n, np.nan)
    
    # Breakout signals
    breakout_above_R1 = (close > R1) & (np.roll(close, 1) <= np.roll(R1, 1))
    breakout_below_S1 = (close < S1) & (np.roll(close, 1) >= np.roll(S1, 1))
    return_to_pivot = (np.abs(close - pivot) < 0.001 * close)  # Within 0.1% of pivot
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 21-period HMA on 1w timeframe
    if len(close_1w) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            weights = np.arange(1, period + 1)
            wma_values = np.convolve(values, weights, mode='valid') / weights.sum()
            # Pad with NaN at beginning
            padded = np.full(len(values), np.nan)
            padded[period-1:] = wma_values
            return padded
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        raw_hma = 2 * wma_half - wma_full
        hma_21 = wma(raw_hma, sqrt_len)
        
        # Trend: price above/below HMA
        hma_rising = np.diff(hma_21, prepend=np.nan) > 0
        hma_falling = np.diff(hma_21, prepend=np.nan) < 0
        price_above_hma = close_1w > hma_21
        price_below_hma = close_1w < hma_21
    else:
        hma_rising = np.full(len(close_1w), False)
        hma_falling = np.full(len(close_1w), False)
        price_above_hma = np.full(len(close_1w), False)
        price_below_hma = np.full(len(close_1w), False)
    
    # Align 1w HMA trend to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    price_above_hma_aligned = align_htf_to_ltf(prices, df_1w, price_above_hma.astype(float))
    price_below_hma_aligned = align_htf_to_ltf(prices, df_1w, price_below_hma.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pivot[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R1 + price above 1w HMA + volume filter
            if (breakout_above_R1[i] and 
                price_above_hma_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below S1 + price below 1w HMA + volume filter
            elif (breakout_below_S1[i] and 
                  price_below_hma_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot OR price below 1w HMA
            if (return_to_pivot[i] or price_below_hma_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot OR price above 1w HMA
            if (return_to_pivot[i] or price_above_hma_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals