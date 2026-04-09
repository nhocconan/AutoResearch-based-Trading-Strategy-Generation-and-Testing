#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + 1h volume spike + 1d ADX regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) for mean reversion entries in ranging markets
# - Confirms with 1h volume > 2.0x 20-period average for institutional participation
# - Filters by 1d ADX < 25 (low trend strength = range-bound market favorable for mean reversion)
# - Exits at opposite Camarilla level (H4/L4) or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) for controlled risk
# - Target: 15-35 trades/year on 4h timeframe (60-140 total over 4 years) to minimize fee drag
# - Works in bull markets (mean reversion in ranges during uptrends) and bear markets (mean reversion in ranges during downtrends)
# - Camarilla levels provide precise support/resistance that adapts to volatility

name = "4h_1h_1d_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1h = get_htf_data(prices, '1h')
    if len(df_1d) < 30 or len(df_1h) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ATR and ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ADX(14) for regime filter
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    dx = np.where((plus_di_14 + minus_di_14) == 0, 0, dx)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d Camarilla levels (based on previous day)
    # H4 = close + 1.5 * (high - low), L4 = close - 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low), L3 = close - 1.25 * (high - low)
    # We use previous day's range to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # 1h volume > 2.0x 20-period average (stricter for fewer trades)
    volume_1h = df_1h['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1h > (2.0 * avg_volume_20)
    
    # Align all HTF indicators to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1h, volume_spike.astype(float))
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or atr_1d_aligned[i] <= 0 or
            adx_1d_aligned[i] >= 25):  # Only trade when ADX < 25 (low trend = range)
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: touch H4 (profit target) or ATR stoploss
            if high[i] >= camarilla_h4_aligned[i]:  # Touch H4 level
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: touch L4 (profit target) or ATR stoploss
            if low[i] <= camarilla_l4_aligned[i]:  # Touch L4 level
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion at Camarilla H3/L3 levels with volume confirmation
            if (low[i] <= camarilla_l3_aligned[i] and   # Touch or break L3 level
                volume_spike_aligned[i]):               # Volume confirmation
                position = 1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            elif (high[i] >= camarilla_h3_aligned[i] and  # Touch or break H3 level
                  volume_spike_aligned[i]):               # Volume confirmation
                position = -1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals