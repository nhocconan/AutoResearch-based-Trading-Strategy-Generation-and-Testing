#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h volume spike + 1d ADX trend filter
# - Uses 1h timeframe for precise entry timing, 4h for volume confirmation, 1d for trend direction
# - Long when: price breaks above H3 Camarilla pivot (1d) AND 4h volume > 2x 20-period average AND 1d ADX > 20
# - Short when: price breaks below L3 Camarilla pivot (1d) AND 4h volume > 2x 20-period average AND 1d ADX > 20
# - Uses discrete position sizing (0.20) to minimize fee churn
# - ATR-based stoploss (2.5x ATR(14)) on 1h timeframe
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we only trade when trend is present

name = "1h_4h_1d_camarilla_adx_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) < 30 or len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot points (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_1d * 1.1 / 4.0)  # H3 = pivot + 1.1*(HL/4)
    l3 = pivot - (range_1d * 1.1 / 4.0)  # L3 = pivot - 1.1*(HL/4)
    h4 = pivot + (range_1d * 1.1 / 2.0)  # H4 = pivot + 1.1*(HL/2)
    l4 = pivot - (range_1d * 1.1 / 2.0)  # L4 = pivot - 1.1*(HL/2)
    
    # Align 1d Camarilla levels to 1h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute 1d ADX(14) for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 4h volume confirmation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)  # Require strong volume spike
    
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1_1h = high_1h - low_1h
    tr2_1h = np.abs(high_1h - np.roll(close_1h, 1))
    tr3_1h = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    tr_1h[0] = tr1_1h[0]
    atr_14 = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_spike_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < L3 Camarilla level OR stoploss hit
            if close_1h[i] < l3_aligned[i] or close_1h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > H3 Camarilla level OR stoploss hit
            if close_1h[i] > h3_aligned[i] or close_1h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume and trend filters
            if vol_spike_aligned[i] and adx_aligned[i] > 20:
                # Long: price > H3 Camarilla level
                if close_1h[i] > h3_aligned[i]:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price < L3 Camarilla level
                elif close_1h[i] < l3_aligned[i]:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals