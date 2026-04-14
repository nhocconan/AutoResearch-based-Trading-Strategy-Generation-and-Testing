#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversals with 1-day trend filter and volume confirmation
# Long when price closes below L3 Camarilla level AND 1d close > 1d open (bullish day) AND volume > 1.5x 20-period average
# Short when price closes above H3 Camarilla level AND 1d close < 1d open (bearish day) AND volume > 1.5x 20-period average
# Exit when price crosses opposite Camarilla level (L3 for longs, H3 for shorts)
# Uses Camarilla pivot levels for mean reversion in ranging markets, daily trend filter for bias, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels on 1d (using previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use L3 and H3 for reversal entries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels (using previous day's data to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first day's previous values to current day's values to avoid NaN
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Calculate Camarilla levels L3 and H3
    camarilla_L3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_H3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    
    # Calculate 1d bullish/bearish bias (close > open = bullish)
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Align daily bias to 4h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_L3_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price below L3 AND bullish daily bias AND volume confirmation
            if (price < camarilla_L3_aligned[i] and daily_bullish_aligned[i] > 0.5 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price above H3 AND bearish daily bias AND volume confirmation
            elif (price > camarilla_H3_aligned[i] and daily_bearish_aligned[i] > 0.5 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back above L3 (mean reversion complete)
            if price > camarilla_L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back below H3 (mean reversion complete)
            if price < camarilla_H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0