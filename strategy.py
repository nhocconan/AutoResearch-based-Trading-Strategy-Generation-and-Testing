#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action within 12h pivot range with volume confirmation
# Long when price rejects 12h S1 support with bullish candle and volume spike
# Short when price rejects 12h R1 resistance with bearish candle and volume spike
# Uses 12h for pivot levels (structure) and 6s for entry timing to avoid whipsaws
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag

name = "6h_P12hRejection_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot levels (higher timeframe structure)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate classic pivot points on 12h OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(pivot_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        open_val = open_[i]
        high_val = high[i]
        low_val = low[i]
        r1_12h_val = r1_12h_aligned[i]
        s1_12h_val = s1_12h_aligned[i]
        pivot_12h_val = pivot_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Candlestick patterns
        bullish_engulfing = i > 0 and close_val > open_[i-1] and open_val < close_[i-1]
        bearish_engulfing = i > 0 and close_val < open_[i-1] and open_val > close_[i-1]
        bullish_hammer = (close_val - low_val) > 2 * (high_val - close_val) and (close_val - open_val) > 0
        bearish_hammer = (high_val - close_val) > 2 * (close_val - low_val) and (open_val - close_val) > 0
        
        if position == 0:
            # Enter long: price near S1 with bullish rejection and volume spike
            near_s1 = low_val <= s1_12h_val * 1.005  # within 0.5% of S1
            bullish_rejection = bullish_engulfing or bullish_hammer
            if near_s1 and bullish_rejection and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price near R1 with bearish rejection and volume spike
            elif high_val >= r1_12h_val * 0.995:  # within 0.5% of R1
                bearish_rejection = bearish_engulfing or bearish_hammer
                if bearish_rejection and vol_spike_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses above pivot or stops bullish momentum
            if close_val > pivot_12h_val * 1.002 or not bullish_rejection:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below pivot or stops bearish momentum
            if close_val < pivot_12h_val * 0.998 or not bearish_rejection:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals