#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot (R3/S3) mean reversion with 1d volatility regime filter.
Long when price touches S3 AND 1d ATR ratio (ATR(5)/ATR(30)) < 0.8 (low volatility regime) AND RSI(14) < 30.
Short when price touches R3 AND 1d ATR ratio < 0.8 AND RSI(14) > 70.
Exit when price reaches the opposite Camarilla level (R3 for longs, S3 for shorts) or midpoint (PP).
Uses 1d HTF for volatility regime to avoid trading during high volatility chop. Target: 50-150 total trades over 4 years (12-37/year).
Camarilla R3/S3 act as intraday support/resistance; mean reversion works best in low volatility regimes.
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
    
    # Calculate 1d ATR ratio for volatility regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_5 / atr_30  # < 0.8 = low volatility regime
    
    # Align 1d ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Camarilla pivot points (based on previous bar's OHLC)
    # Camarilla levels: R4 = PP + (H-L)*1.1/2, R3 = PP + (H-L)*1.1/4, etc.
    # We use previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r3 = pp + range_hl * 1.1 / 4
    s3 = pp - range_hl * 1.1 / 4
    
    # 6h RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 14)  # ATR30 (30), RSI (14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(pp[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_regime = atr_ratio_aligned[i]
        rsi_val = rsi[i]
        r3_val = r3[i]
        s3_val = s3[i]
        pp_val = pp[i]
        
        if position == 0:
            # Long: Price at S3 AND low volatility regime AND RSI oversold
            if abs(price - s3_val) < 0.001 * price and vol_regime < 0.8 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: Price at R3 AND low volatility regime AND RSI overbought
            elif abs(price - r3_val) < 0.001 * price and vol_regime < 0.8 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:  # Long
                # Exit at R3 (profit target) or PP (midpoint)
                if price >= r3_val or price >= pp_val:
                    exit_signal = True
            elif position == -1:  # Short
                # Exit at S3 (profit target) or PP (midpoint)
                if price <= s3_val or price <= pp_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_MeanReversion_1dATRratio_VolRegime_RSI_LevelExit"
timeframe = "6h"
leverage = 1.0