#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
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
    
    # Camarilla levels: R4, R3, S3, S4
    r4 = pivot + (range_1d * 1.1 / 2)
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed daily bars
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    r4[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align daily levels to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h ATR for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
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
        
        # Long conditions: price breaks above S3 or S4 with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_low < s3_12h[i] or price_low < s4_12h[i])
        
        # Short conditions: price breaks below R3 or R4 with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_high > r3_12h[i] or price_high > r4_12h[i])
        
        # Exit when price returns to pivot level
        pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
        exit_long = position == 1 and price_close > pivot_12h[i]
        exit_short = position == -1 and price_close < pivot_12h[i]
        
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

# Hypothesis: Camarilla pivot levels from daily timeframe provide key support/resistance levels for 12h trading.
# Enters long when price breaks below S3/S4 (oversold bounce) with volume confirmation (>1.5x average) and sufficient volatility.
# Enters short when price breaks above R3/R4 (overbought rejection) with same conditions.
# Exits when price returns to the daily pivot level, capturing mean reversion in ranging markets.
# Works in both bull and bear markets by fading extremes at key pivot levels.
# Volume confirmation ensures institutional participation at these key levels.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe.
# Uses Camarilla formula for mathematically derived support/resistance levels.
# Volatility filter prevents whipsaws in low-volatility environments.