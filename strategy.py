#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Long when price breaks above Camarilla H3 (1d) AND 1d volume > 1.5x 20-period mean AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 (1d) AND 1d volume > 1.5x 20-period mean AND chop < 61.8 (trending)
# - Exit when price returns to Camarilla Pivot point (1d) or opposite signal
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets via breakout continuation, in bear markets via breakdown continuation

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Choppiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = np.abs(high[i] - close[i-1])
            lc = np.abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close)
        for i in range(window-1, len(close)):
            if atr[i] > 0 and hh[i] > ll[i]:
                log_sum = np.log(atr[i] * window) / np.log(2)
                log_range = np.log((hh[i] - ll[i]) / atr[i]) / np.log(2)
                chop[i] = 100 * (log_sum / log_range)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop < 61.8  # Trending regime
    
    # Pre-compute 4h volume spike (volume > 1.5x 20-period mean)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_h3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND trending regime
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike[i] and chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND trending regime
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike[i] and chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to pivot level or opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] <= camarilla_pivot_aligned[i] or
                         close[i] < camarilla_l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] >= camarilla_pivot_aligned[i] or
                          close[i] > camarilla_h3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals