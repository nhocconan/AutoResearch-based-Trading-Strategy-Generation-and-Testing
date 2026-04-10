#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND chop > 61.8
# - Exit when price crosses Camarilla H4/L4 levels (strong reversal) or chop < 38.2 (trend regime)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we trade in ranging markets where mean reversion works

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Camarilla pivot levels (based on previous day)
    def calculate_camarilla(h, l, c):
        # Camarilla levels based on previous day's OHLC
        range_val = h - l
        h5 = c + (range_val * 1.1 / 2)
        h4 = c + (range_val * 1.1 / 4)
        h3 = c + (range_val * 1.1 / 6)
        l3 = c - (range_val * 1.1 / 6)
        l4 = c - (range_val * 1.1 / 4)
        l5 = c - (range_val * 1.1 / 2)
        return h3, h4, h5, l3, l4, l5
    
    # Calculate Camarilla levels for each 12h bar using previous 1d bar
    h3_12h = np.full(n, np.nan)
    h4_12h = np.full(n, np.nan)
    l3_12h = np.full(n, np.nan)
    l4_12h = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 1d bar's OHLC (need to map 12h index to 1d index)
        # For simplicity, use rolling window on 12h data with 2-period lookback
        if i >= 2:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            h3, h4, h5, l3, l4, l5 = calculate_camarilla(prev_high, prev_low, prev_close)
            h3_12h[i] = h3
            h4_12h[i] = h4
            l3_12h[i] = l3
            l4_12h[i] = l4
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h Choppiness Index (CHOP) for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR (Wilder's smoothing)
        atr = np.zeros_like(tr)
        atr[window-1] = np.mean(tr[1:window])  # First ATR value
        for i in range(window, len(tr)):
            atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        
        # Sum of ATR over window
        atr_sum = np.zeros_like(atr)
        for i in range(window-1, len(atr)):
            atr_sum[i] = np.sum(atr[i-window+1:i+1])
        
        # Highest high and lowest low over window
        hh = np.zeros_like(high_arr)
        ll = np.zeros_like(low_arr)
        for i in range(window-1, len(high_arr)):
            hh[i] = np.max(high_arr[i-window+1:i+1])
            ll[i] = np.min(low_arr[i-window+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close_arr)
        for i in range(window-1, len(close_arr)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50  # Neutral when no range
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_range = chop > 61.8  # Range regime (mean revert)
    chop_trend = chop < 38.2  # Trend regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND range regime
            if (close[i] > h3_12h[i] and 
                volume_spike[i] and 
                chop_range[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND range regime
            elif (close[i] < l3_12h[i] and 
                  volume_spike[i] and 
                  chop_range[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses H4/L4 (strong reversal) OR trend regime begins
            exit_long = (position == 1 and (close[i] > h4_12h[i] or chop_trend[i]))
            exit_short = (position == -1 and (close[i] < l4_12h[i] or chop_trend[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals