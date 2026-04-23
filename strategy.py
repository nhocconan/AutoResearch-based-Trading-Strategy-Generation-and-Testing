#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 mean reversion with 1d volume regime filter.
Long when price touches S3 level AND 1d volume > 1.5x 20-period average AND RSI(14) < 30.
Short when price touches R3 level AND 1d volume > 1.5x 20-period average AND RSI(14) > 70.
Exit when price reaches opposite Camarilla level (R3 for longs, S3 for shorts) or midpoint (P).
Uses 1d HTF for volume regime to avoid low-liquidity false signals. Target: 50-150 total trades over 4 years (12-37/year).
Camarilla levels provide precise intraday support/resistance; volume regime ensures institutional participation; RSI filter confirms exhaustion.
Works in both bull (buy dips at S3) and bear (sell rallies at R3) markets when volume confirms institutional interest.
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
    
    # Calculate 1d volume regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h RSI for momentum exhaustion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # need 20 bars for Camarilla calculation (based on prior day)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for 6h bar using prior 6h bar's OHLC
        # Camarilla uses previous period's range
        if i == 0:
            # Need previous bar data, skip first bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        range_val = phigh - plow
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla levels
        R3 = pclose + range_val * 1.1 / 4
        S3 = pclose - range_val * 1.1 / 4
        R4 = pclose + range_val * 1.1 / 2
        S4 = pclose - range_val * 1.1 / 2
        P = (phigh + plow + pclose) / 3  # pivot point
        
        price = close[i]
        vol_regime = vol_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: Price at S3 support AND high volume regime AND RSI oversold
            if abs(price - S3) < 0.001 * price and vol_regime > 1.5 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: Price at R3 resistance AND high volume regime AND RSI overbought
            elif abs(price - R3) < 0.001 * price and vol_regime > 1.5 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Long exit: price reaches R3 (resistance) or P (pivot)
                if price >= R3 * 0.999 or price >= P * 0.999:
                    exit_signal = True
            elif position == -1:
                # Short exit: price reaches S3 (support) or P (pivot)
                if price <= S3 * 1.001 or price <= P * 1.001:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_MeanReversion_1dVolumeRegime_RSIExhaustion"
timeframe = "6h"
leverage = 1.0