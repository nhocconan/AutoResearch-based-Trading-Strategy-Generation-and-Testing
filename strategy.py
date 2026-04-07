#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price touches Camarilla L3 level with bullish rejection (close > open) and volume > 1.5x average, enter short when price touches H3 level with bearish rejection (close < open) and volume > 1.5x average. Uses 1d trend filter (price above/below 200-period EMA) to avoid counter-trend trades. Designed for 15-25 trades/year to minimize fee drag while capturing mean-reversion moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(vol_ma[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or
            np.isnan(open_price[i]) or np.isnan(high[i-1]) or np.isnan(low[i-1]) or
            np.isnan(open_price[i-1]) or np.isnan(close[i-1])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        # Calculate Camarilla levels from previous day
        # Camarilla levels: based on previous day's range
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate pivot and range
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        L3 = prev_close - (range_val * 1.1 / 4)
        H3 = prev_close + (range_val * 1.1 / 4)
        L4 = prev_close - (range_val * 1.1 / 2)
        H4 = prev_close + (range_val * 1.1 / 2)
        
        # Price action signals
        bullish_rejection = close[i] > open_price[i]  # Bullish candle
        bearish_rejection = close[i] < open_price[i]  # Bearish candle
        
        if position == 1:  # Long position
            # Exit: price closes below L4 (break of support)
            if close[i] < L4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H4 (break of resistance)
            if close[i] > H4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches L3 with bullish rejection + price > 1d EMA200
                if (abs(close[i] - L3) < (range_val * 0.05) and  # Within 5% of L3
                    bullish_rejection and 
                    close[i] > ema_200_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H3 with bearish rejection + price < 1d EMA200
                elif (abs(close[i] - H3) < (range_val * 0.05) and  # Within 5% of H3
                      bearish_rejection and 
                      close[i] < ema_200_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals