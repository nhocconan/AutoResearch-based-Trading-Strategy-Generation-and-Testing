#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-day Camarilla pivot levels (S1/R1) for mean-reversion entries in ranging markets.
In ranging conditions (Choppiness Index > 61.8), buy at S1 with volume confirmation, sell at R1 with volume confirmation.
Uses 1-day structure for pivot calculation (more reliable than intraday) and 12h for execution.
Trades only during high-liquidity sessions (UTC 8:00-20:00) to avoid low-volume false signals.
Targets 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (use prior day to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: S1, R1 based on previous day
    # Camarilla formulas:
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    range_1d = high_1d - low_1d
    s1 = close_1d - (range_1d * 1.1 / 12)
    r1 = close_1d + (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Choppiness Index regime filter (14-period) - calculated on 12h data
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes over 14 periods
    abs_close_change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    sum_abs_change = pd.Series(abs_close_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum_abs_change / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_abs_change / (atr_14 * 14)) / np.log10(14)
    
    # Volume confirmation (volume > 1.2x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # Session filter: UTC 8:00-20:00 (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        chop_val = chop[i]
        vol_ratio_val = vol_ratio[i]
        in_sess = in_session[i]
        
        # Regime filter: only trade in ranging markets (Choppiness > 61.8)
        if chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0 and in_sess:
            # Enter long at S1 with volume confirmation
            if (price_close <= s1_val * 1.002 and  # Allow small buffer for slippage
                vol_ratio_val > 1.2):
                signals[i] = 0.25
                position = 1
            # Enter short at R1 with volume confirmation
            elif (price_close >= r1_val * 0.998 and  # Allow small buffer for slippage
                  vol_ratio_val > 1.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Long exit: price reaches midpoint (mean reversion target) or stop loss
                midpoint = (s1_val + r1_val) / 2
                if price_close >= midpoint:
                    signals[i] = 0.0
                    position = 0
                # Stop loss: close below S1 with confirmation
                elif price_close < s1_val * 0.995:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price reaches midpoint or stop loss
                midpoint = (s1_val + r1_val) / 2
                if price_close <= midpoint:
                    signals[i] = 0.0
                    position = 0
                # Stop loss: close above R1 with confirmation
                elif price_close > r1_val * 1.005:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_S1R1_MeanReversion_Chop_Volume"
timeframe = "12h"
leverage = 1.0