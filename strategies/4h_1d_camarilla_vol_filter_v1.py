#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and volatility filter
# Uses 1d Camarilla pivot levels (H3/L3) from previous day for breakout signals
# Enters only when 1d ATR rank < 30 (low volatility) to avoid whipsaws in choppy markets
# Volume confirmation: 4h volume > 1.5x 20-period average (~5 days)
# Exits when price closes opposite Camarilla level (H4/L4)
# Position size 0.25 to limit drawdown
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag
# Works in both bull/bear: Camarilla provides structure, low vol filter avoids false breakouts

name = "4h_1d_camarilla_vol_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h5 = np.full(len(df_1d), np.nan)
    camarilla_l5 = np.full(len(df_1d), np.nan)
    camarilla_h6 = np.full(len(df_1d), np.nan)
    camarilla_l6 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        pivot = (phigh + plow + pclose) / 3
        range_val = phigh - plow
        
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
        camarilla_h5[i] = pclose + range_val * 1.1
        camarilla_l5[i] = pclose - range_val * 1.1
        camarilla_h6[i] = pclose + range_val * 1.1 * 1.166
        camarilla_l6[i] = pclose - range_val * 1.1 * 1.166
    
    # Calculate 1d ATR (14-period)
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # ATR percentile rank (100-day lookback ~ 3 months)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(100, len(df_1d)):
        window = atr_1d[i-100:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align 1d data to 4h timeframe (only use completed daily bars)
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_rank_4h = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Volume confirmation: 20-period average on 4h (~5 days)
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after ATR rank warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_4h[i]) or 
            np.isnan(camarilla_l3_4h[i]) or 
            np.isnan(camarilla_h4_4h[i]) or 
            np.isnan(camarilla_l4_4h[i]) or 
            np.isnan(atr_rank_4h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility environment (ATR rank < 30 = bottom 30% volatility)
        if atr_rank_4h[i] >= 30:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d Camarilla L4
            if close[i] <= camarilla_l4_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d Camarilla H4
            if close[i] >= camarilla_h4_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 1d Camarilla H3 with volume confirmation
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > camarilla_h3_4h[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 1d Camarilla L3 with volume confirmation
            elif (close[i] < camarilla_l3_4h[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals