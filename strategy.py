#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels + 12h volume spike + 12h ADX trend filter
# - Primary signal: Price touches Camarilla H3/L3 levels from prior 12h session
# - Volume confirmation: 12h volume > 1.3x 24-period average volume (ensures participation)
# - Trend filter: 12h ADX > 25 (trending market) enables breakout continuation
# - In strong trends (ADX > 25): breakouts continue; in weak trends (ADX < 20): mean reversion at H4/L4
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "4h_12h_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume spike filter
    volume_12h = df_12h['volume'].values
    avg_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume_12h > (1.3 * avg_volume_24)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Pre-compute 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 12h Camarilla levels (based on prior 12h bar's OHLC)
    # Camarilla: H4 = C + 1.1*(H-L)*1.1/2, L4 = C - 1.1*(H-L)*1.1/2
    #            H3 = C + 1.1*(H-L)*1.1/4, L3 = C - 1.1*(H-L)*1.1/4
    #            H2 = C + 1.1*(H-L)*1.1/6, L2 = C - 1.1*(H-L)*1.1/6
    #            H1 = C + 1.1*(H-L)*1.1/12, L1 = C - 1.1*(H-L)*1.1/12
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_h3 = np.zeros_like(close_12h)
    camarilla_l3 = np.zeros_like(close_12h)
    camarilla_h4 = np.zeros_like(close_12h)
    camarilla_l4 = np.zeros_like(close_12h)
    
    for i in range(len(close_12h)):
        if i == 0:
            camarilla_h3[i] = camarilla_l3[i] = camarilla_h4[i] = camarilla_l4[i] = close_12h[i]
        else:
            rng = high_12h[i-1] - low_12h[i-1]
            camarilla_h3[i] = close_12h[i-1] + 1.1 * rng * 1.1 / 4
            camarilla_l3[i] = close_12h[i-1] - 1.1 * rng * 1.1 / 4
            camarilla_h4[i] = close_12h[i-1] + 1.1 * rng * 1.1 / 2
            camarilla_l4[i] = close_12h[i-1] - 1.1 * rng * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Pre-compute 4h ATR(14) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: mean reversion at H4 OR stoploss hit
            if close_4h[i] >= camarilla_h4_aligned[i] * 0.999 or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: mean reversion at L4 OR stoploss hit
            if close_4h[i] <= camarilla_l4_aligned[i] * 1.001 or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume spike and ADX filter
            # In strong trends (ADX > 25): breakout continuation from H3/L3
            # In weak trends (ADX < 20): mean reversion at H4/L4
            if volume_spike_aligned[i]:
                if adx_aligned[i] > 25:  # strong trend - breakout continuation
                    # Long: price breaks above H3
                    if close_4h[i] > camarilla_h3_aligned[i]:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price breaks below L3
                    elif close_4h[i] < camarilla_l3_aligned[i]:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
                elif adx_aligned[i] < 20:  # weak trend - mean reversion
                    # Long: price touches L4 (support)
                    if close_4h[i] <= camarilla_l4_aligned[i] * 1.001:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price touches H4 (resistance)
                    elif close_4h[i] >= camarilla_h4_aligned[i] * 0.999:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
    
    return signals