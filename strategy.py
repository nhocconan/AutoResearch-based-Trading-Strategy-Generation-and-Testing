#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d ADX trend filter
# - Uses 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# - Breaks above R4 or below S4 with 12h volume > 1.5x 24-period average signal continuation
# - Trades only when 1d ADX > 25 (trending market) to avoid false breakouts in ranging markets
# - Position size: 0.25 (25% of capital) to manage drawdown in volatile 6h timeframe
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Camarilla levels provide mathematical support/resistance that adapts to volatility

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(10) for ADX calculation
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # 1d ADX(14) for trend strength
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = np.where(tr_14 > 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di = np.where(tr_14 > 0, (minus_dm_14 / tr_14) * 100, 0)
    dx = np.where((plus_di + minus_di) > 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d Camarilla pivot levels (based on previous day)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Pre-compute 12h indicators
    volume_12h = df_12h['volume'].values
    close_12h = df_12h['close'].values
    
    # 12h volume > 1.5x 24-period average (confirm institutional interest)
    avg_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume_12h > (1.5 * avg_volume_24)
    
    # Align all HTF indicators to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            adx_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price touches S3 (mean reversion level)
            if low[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches R3 (mean reversion level)
            if high[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout beyond R4/S4 with volume confirmation and ADX trend filter
            if (high[i] >= r4_aligned[i] and    # Break above R4
                volume_spike_aligned[i] and     # Volume confirmation
                adx_1d_aligned[i] > 25):        # Strong trend (ADX > 25)
                position = 1
                entry_price = high[i]
                signals[i] = 0.25
            elif (low[i] <= s4_aligned[i] and   # Break below S4
                  volume_spike_aligned[i] and   # Volume confirmation
                  adx_1d_aligned[i] > 25):      # Strong trend (ADX > 25)
                position = -1
                entry_price = low[i]
                signals[i] = -0.25
    
    return signals