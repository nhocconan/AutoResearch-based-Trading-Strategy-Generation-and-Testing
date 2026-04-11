#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_reversion_v1"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1, S2, S3, S4, R1, R2, R3, R4
    s1 = pivot - (range_1d * 1.1 / 12)
    s2 = pivot - (range_1d * 1.1 / 6)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    r1 = pivot + (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed daily bars
    s1 = np.roll(s1, 1); s2 = np.roll(s2, 1); s3 = np.roll(s3, 1); s4 = np.roll(s4, 1)
    r1 = np.roll(r1, 1); r2 = np.roll(r2, 1); r3 = np.roll(r3, 1); r4 = np.roll(r4, 1)
    s1[0] = s2[0] = s3[0] = s4[0] = r1[0] = r2[0] = r3[0] = r4[0] = np.nan
    
    # Align daily levels to 4h timeframe
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    
    # Calculate 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(s1_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(r1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.005 * price_close  # ATR > 0.5% of price
        
        # Long conditions: price closes below S3 (oversold bounce) with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_close < s3_4h[i])
        
        # Short conditions: price closes above R3 (overbought rejection) with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_close > r3_4h[i])
        
        # Exit when price returns to daily S2 (long) or R2 (short)
        exit_long = position == 1 and price_close > s2_4h[i]
        exit_short = position == -1 and price_close < r2_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily Camarilla S3/R3 levels act as strong support/resistance for 4h price reversals.
# Enters long when 4h price closes below S3 (oversold) with volume confirmation (>1.5x average) and sufficient volatility (ATR > 0.5% of price).
# Enters short when 4h price closes above R3 (overbought) with same conditions.
# Exits long when price returns above S2 (mean reversion to middle of range).
# Exits short when price returns below R2 (mean reversion to middle of range).
# Uses only close prices for entry/exit to avoid look-ahead.
# Designed for 30-60 trades per year to minimize fee drag on 4h timeframe.
# Works in both bull (buying dips at S3) and bear (selling rallies at R3) markets.