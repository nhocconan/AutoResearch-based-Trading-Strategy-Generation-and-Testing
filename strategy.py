#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.3x 20-period average AND daily ATR(14) > 0.5 * 20-period ATR mean
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.3x 20-period average AND daily ATR(14) > 0.5 * 20-period ATR mean
# - Exit when price crosses Camarilla H4/L4 levels (strong reversal) or ATR regime filter fails
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots provide intraday support/resistance; volume confirms breakout validity
# - ATR regime filter ensures sufficient volatility for meaningful moves
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Camarilla pivot levels (based on previous day)
    def calculate_camarilla(h_prev, l_prev, c_prev):
        range_prev = h_prev - l_prev
        H4 = c_prev + range_prev * 1.1 / 2
        H3 = c_prev + range_prev * 1.1 / 4
        H2 = c_prev + range_prev * 1.1 / 6
        H1 = c_prev + range_prev * 1.1 / 12
        L1 = c_prev - range_prev * 1.1 / 12
        L2 = c_prev - range_prev * 1.1 / 6
        L3 = c_prev - range_prev * 1.1 / 4
        L4 = c_prev - range_prev * 1.1 / 2
        return H3, L3, H4, L4
    
    # Shift previous day's OHLC to align with current 12h bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_H3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_L3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_H4 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_L4 = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(1, len(close_1d)):
        H3, L3, H4, L4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_H3[i] = H3
        camarilla_L3[i] = L3
        camarilla_H4[i] = H4
        camarilla_L4[i] = L4
    
    # Pre-compute 12h ATR (14-period) for regime filter
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 12h ATR mean (20-period) for regime filter threshold
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    atr_mean_20 = rolling_mean(atr, 20)
    atr_regime = atr > (0.5 * atr_mean_20)  # ATR regime filter: current ATR > 50% of 20-period mean
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Align HTF indicators to 12h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1d volume > 1.3x 20-period average)
        vol_spike = volume_1d[i] > 1.3 * vol_ma_1d_aligned[i] if not np.isnan(vol_ma_1d_aligned[i]) else False
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Camarilla H3 breakout AND volume spike AND ATR regime
            if (close[i] > camarilla_H3_aligned[i] and vol_spike and atr_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla L3 breakdown AND volume spike AND ATR regime
            elif (close[i] < camarilla_L3_aligned[i] and vol_spike and atr_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla H4/L4 levels (strong reversal)
            exit_long = (position == 1 and close[i] > camarilla_H4_aligned[i])
            exit_short = (position == -1 and close[i] < camarilla_L4_aligned[i])
            
            # Optional: exit if ATR regime filter fails (low volatility environment)
            regime_exit = not atr_regime[i]
            
            if exit_long or exit_short or regime_exit:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result