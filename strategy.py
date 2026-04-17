#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above R1 AND volume > 1.8x 20-period average AND price > 1d EMA50 (bullish regime).
Short when price breaks below S1 AND volume > 1.8x 20-period average AND price < 1d EMA50 (bearish regime).
Exit when price crosses the 1d EMA50 in opposite direction.
Camarilla levels provide intraday support/resistance, 1d EMA50 filters higher timeframe trend,
volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
by trading with the 1d trend while using Camarilla for precise entry/exit.
Targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    open_4h = df_4h['open'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels on 1d timeframe (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align all indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 1.8x avg AND price > 1d EMA50 (bullish regime)
            if high_price > r1 and vol > 1.8 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 1.8x avg AND price < 1d EMA50 (bearish regime)
            elif low_price < s1 and vol > 1.8 * vol_ma and price < ema_50:
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

name = "4h_Camarilla_R1S1_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0