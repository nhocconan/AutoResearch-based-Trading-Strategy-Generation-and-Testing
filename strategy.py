#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Uses 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# - Enters long when price breaks above R4 with 12h volume > 1.5x 20-period average
# - Enters short when price breaks below S4 with 12h volume > 1.5x 20-period average
# - Filters by 1d ADX > 25 to ensure trending market (avoids chop)
# - Exits when price touches opposite Camarilla level (R3/S3) or 12h ATR-based stop (2.0x ATR)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag
# - Camarilla pivots provide mathematically derived support/resistance that works in all regimes
# - Breakouts at R4/S4 with volume confirmation capture strong moves
# - ADX filter ensures we only trade in trending markets, avoiding false breakouts in chop

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0] if len(tr_12h) > 0 else 0
    
    # 12h ATR(14) for stoploss
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    prev_close_12h[0] = close_12h[0]
    
    camarilla_range = prev_high_12h - prev_low_12h
    camarilla_range = np.where(camarilla_range <= 0, 1e-10, camarilla_range)
    
    camarilla_r4 = prev_close_12h + 1.5 * camarilla_range
    camarilla_r3 = prev_close_12h + 1.1 * camarilla_range
    camarilla_s3 = prev_close_12h - 1.1 * camarilla_range
    camarilla_s4 = prev_close_12h - 1.5 * camarilla_range
    
    # 12h Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0] if len(tr_1d) > 0 else 0
    
    # 1d ATR(14) for ADX
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d +DM and -DM for ADX
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # 1d smoothed +DM, -DM, ATR for ADX(14)
    atr_14_smooth = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # 1d ADX(14)
    plus_di = 100 * plus_dm_smooth / np.where(atr_14_smooth > 0, atr_14_smooth, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.where(atr_14_smooth > 0, atr_14_smooth, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or atr_12h_aligned[i] <= 0 or adx_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Camarilla touch (S3) or ATR stoploss
            if low[i] <= camarilla_s3_aligned[i]:  # Touch S3 level
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla touch (R3) or ATR stoploss
            if high[i] >= camarilla_r3_aligned[i]:  # Touch R3 level
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and trend filter
            if (high[i] >= camarilla_r4_aligned[i] and  # Break above R4
                volume_spike_aligned[i] and         # Volume confirmation
                adx_1d_aligned[i] > 25):            # Trending market (ADX > 25)
                position = 1
                entry_price = high[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= camarilla_s4_aligned[i] and   # Break below S4
                  volume_spike_aligned[i] and         # Volume confirmation
                  adx_1d_aligned[i] > 25):            # Trending market (ADX > 25)
                position = -1
                entry_price = low[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = -0.25
    
    return signals