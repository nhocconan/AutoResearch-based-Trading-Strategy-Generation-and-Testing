#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with Volume Spike and Daily Trend Filter
# Uses daily Camarilla pivot levels (H4/L4) from prior day + volume confirmation + daily EMA50 trend filter
# Camarilla levels provide precise intraday support/resistance, volume confirms breakout validity
# Daily EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
# Works in bull/bear by only taking breakouts in direction of daily trend
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels for each day (using prior day's OHLC)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # We'll use prior day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Calculate volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Align daily indicators to 4h
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price breaks above Camarilla H4 with volume spike and above daily EMA50
            if (price > camarilla_h4_aligned[i] and vol_spike[i] and 
                price > ema_50_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short breakout: price breaks below Camarilla L4 with volume spike and below daily EMA50
            elif (price < camarilla_l4_aligned[i] and vol_spike[i] and 
                  price < ema_50_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Camarilla L4 or below daily EMA50
            if price < camarilla_l4_aligned[i] or price < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Camarilla H4 or above daily EMA50
            if price > camarilla_h4_aligned[i] or price > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_EMA50"
timeframe = "4h"
leverage = 1.0