#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND volume > 2.0x 20-period average AND price > 12h EMA34.
Short when price breaks below S1 AND volume > 2.0x 20-period average AND price < 12h EMA34.
Exit when price crosses the 12h EMA34 in opposite direction.
Camarilla pivot levels provide high-probability intraday support/resistance, 12h EMA34 filters for
medium-term trend, volume confirmation reduces false breakouts. Designed for low trade frequency
(20-50/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels on 4h timeframe (using previous day's OHLC)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need previous 4h bar's OHLC to calculate today's levels
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    # Set first value to NaN (no previous bar)
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    camarilla_upper = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    camarilla_lower = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_4h, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_4h, camarilla_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_upper_aligned[i]) or 
            np.isnan(camarilla_lower_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_34 = ema_34_12h_aligned[i]
        r1 = camarilla_upper_aligned[i]
        s1 = camarilla_lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 2.0x avg AND price > 12h EMA34 (bullish trend)
            if high_price > r1 and vol > 2.0 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 2.0x avg AND price < 12h EMA34 (bearish trend)
            elif low_price < s1 and vol > 2.0 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h EMA34
            if price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA34
            if price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA34_Filter"
timeframe = "4h"
leverage = 1.0