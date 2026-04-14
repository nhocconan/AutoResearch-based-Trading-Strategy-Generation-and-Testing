#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day pivot points with range-bound and breakout conditions.
# Long when price breaks above R3 with volume and price > 1-day VWAP (bullish continuation).
# Short when price breaks below S3 with volume and price < 1-day VWAP (bearish continuation).
# Fade trades: long at S1 with price < VWAP and RSI < 30, short at R1 with price > VWAP and RSI > 70.
# Uses 1-day pivot levels (classic formula) for structure, VWAP for intraday bias, and volume for confirmation.
# Designed to work in both trending and ranging markets by adapting to price action relative to pivot levels.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for pivot levels and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP
    vwap_1d = (close_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d[volume_1d.cumsum() == 0] = np.nan
    
    # Calculate classic pivot points: P = (H + L + C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1 = 2 * pivot - high_1d
    s2 = pivot - (high_1d - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    r1 = 2 * pivot - low_1d
    r2 = pivot + (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    
    # Load 1d RSI for fade filter confirmation
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align all indicators to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need VWAP and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Price relative to VWAP for bias
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        if position == 0:
            # Breakout continuation trades
            # Long: break above R3 with volume and price > VWAP
            if (close[i] > r3_aligned[i] and 
                volume_confirmed and 
                price_above_vwap):
                position = 1
                signals[i] = position_size
            # Short: break below S3 with volume and price < VWAP
            elif (close[i] < s3_aligned[i] and 
                  volume_confirmed and 
                  price_below_vwap):
                position = -1
                signals[i] = -position_size
            # Fade trades at S1/R1
            # Long: at S1 with price < VWAP and oversold RSI
            elif (close[i] <= s1_aligned[i] and 
                  price_below_vwap and 
                  rsi_1d_aligned[i] < 30):
                position = 1
                signals[i] = position_size
            # Short: at R1 with price > VWAP and overbought RSI
            elif (close[i] >= r1_aligned[i] and 
                  price_above_vwap and 
                  rsi_1d_aligned[i] > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or RSI > 70 (overbought)
            if (close[i] <= pivot_aligned[i] or 
                rsi_1d_aligned[i] >= 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot or RSI < 30 (oversold)
            if (close[i] >= pivot_aligned[i] or 
                rsi_1d_aligned[i] <= 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dPivot_RangeBreakout_Fade_v1"
timeframe = "6h"
leverage = 1.0