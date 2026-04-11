#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for each day
    H = high_1d
    L = low_1d
    C = close_1d
    R = H - L
    
    # Resistance levels
    R4 = C + (R * 1.5 / 2)
    R3 = C + (R * 1.25 / 2)
    R2 = C + (R * 1.1 / 2)
    R1 = C + (R * 1.05 / 2)
    
    # Support levels
    S1 = C - (R * 1.05 / 2)
    S2 = C - (R * 1.1 / 2)
    S3 = C - (R * 1.25 / 2)
    S4 = C - (R * 1.5 / 2)
    
    # Shift by 1 to use only completed daily bars
    R4 = np.roll(R4, 1)
    R3 = np.roll(R3, 1)
    R2 = np.roll(R2, 1)
    R1 = np.roll(R1, 1)
    S1 = np.roll(S1, 1)
    S2 = np.roll(S2, 1)
    S3 = np.roll(S3, 1)
    S4 = np.roll(S4, 1)
    R4[0] = np.nan
    R3[0] = np.nan
    R2[0] = np.nan
    R1[0] = np.nan
    S1[0] = np.nan
    S2[0] = np.nan
    S3[0] = np.nan
    S4[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 4h Donchian channel (20-period) for trend filter
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(R4_4h[i]) or np.isnan(R3_4h[i]) or np.isnan(R2_4h[i]) or np.isnan(R1_4h[i]) or
            np.isnan(S1_4h[i]) or np.isnan(S2_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(S4_4h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price above/below Donchian midpoint
        donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
        uptrend = price_close > donchian_mid
        downtrend = price_close < donchian_mid
        
        # Long conditions: price breaks above R3 with volume and uptrend
        long_breakout = price_high > R3_4h[i]
        long_signal = volume_confirmed and long_breakout and uptrend
        
        # Short conditions: price breaks below S3 with volume and downtrend
        short_breakout = price_low < S3_4h[i]
        short_signal = volume_confirmed and short_breakout and downtrend
        
        # Exit when price returns to Donchian midpoint
        exit_long = position == 1 and price_close < donchian_mid
        exit_short = position == -1 and price_close > donchian_mid
        
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

# Hypothesis: Camarilla breakout with volume confirmation and Donchian trend filter on 4h.
# Uses daily Camarilla levels (R3/S3) as key support/resistance. Enters long when price
# breaks above R3 with volume confirmation (>1.5x average) and uptrend (price > Donchian midpoint).
# Enters short when price breaks below S3 with volume and downtrend (price < Donchian midpoint).
# Exits when price returns to Donchian midpoint. Works in both bull and bear markets by
# aligning with the higher timeframe daily structure and trend filter. Target: 75-200 total
# trades over 4 years (19-50/year) to minimize fee drag on 4h timeframe. Camarilla levels
# provide institutional-grade support/resistance, reducing false breakouts. Volume confirmation
# ensures institutional participation. Trend filter prevents counter-trend trades.