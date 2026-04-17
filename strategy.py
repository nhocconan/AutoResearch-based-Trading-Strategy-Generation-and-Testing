#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above R1 AND volume > 2.0x 20-period average AND price > 1d EMA50.
Short when price breaks below S1 AND volume > 2.0x 20-period average AND price < 1d EMA50.
Exit when price crosses the 1d EMA50 in opposite direction.
Camarilla pivot levels provide high-probability support/resistance, 1d EMA50 filters for
longer-term trend, volume confirmation reduces false breakouts. Designed for low trade frequency
(12-37/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1d timeframe (using previous day's OHLC)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need previous 1d bar's OHLC to calculate today's levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # Set first value to NaN (no previous bar)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_upper = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_lower = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Calculate volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1d, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_upper_aligned[i]) or 
            np.isnan(camarilla_lower_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        r1 = camarilla_upper_aligned[i]
        s1 = camarilla_lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 2.0x avg AND price > 1d EMA50 (bullish trend)
            if high_price > r1 and vol > 2.0 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 2.0x avg AND price < 1d EMA50 (bearish trend)
            elif low_price < s1 and vol > 2.0 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA50
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA50
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1dEMA50_Filter"
timeframe = "12h"
leverage = 1.0