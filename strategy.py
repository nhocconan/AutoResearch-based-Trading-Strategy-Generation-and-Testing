#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d with volume confirmation and choppiness filter
# - Long: price breaks above R4 with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market)
# - Short: price breaks below S4 with volume > 1.5x 20-period average and CHOP > 61.8
# - Exit: price returns to R3/S3 levels (mean reversion)
# - Uses 1d Camarilla levels from prior day OHLC, aligned to 12h
# - Choppiness filter avoids trending markets where breakouts fail
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by fading extremes in ranging conditions

name = "12h_1d_camarilla_breakout_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels for current day)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index (14-period) on 12h timeframe
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Choppiness filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop[i] > 61.8
        
        # Price position relative to Camarilla levels
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with volume and chop filter
        if close_price > r4 and vol_confirm and chop_filter:
            enter_long = True
        
        # Short breakout: price breaks below S4 with volume and chop filter
        if close_price < s4 and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: mean reversion at R3/S3 levels
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to R3
            exit_long = close_price <= r3
        elif position == -1:
            # Exit short if price rises back to S3
            exit_short = close_price >= s3
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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