#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Uses 12h Camarilla pivot levels (H3/L3) for breakout entries
# - Requires 12h volume > 1.5x 20-period average for confirmation
# - Uses 4h choppiness index (CHOP > 61.8) to avoid trending markets (mean reversion in range)
# - Only takes long when price breaks above H3, short when breaks below L3
# - Exit when price reverts to 12h pivot point (PP) or opposite Camarilla level
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position size: 0.25 (25% of capital) discrete levels to minimize fee churn
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h OHLC for Camarilla pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot Point (PP) = (H + L + C) / 3
    # Resistance/H3 = C + (H - L) * 1.1 / 2
    # Support/L3 = C - (H - L) * 1.1 / 2
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    h3_12h = close_12h + (range_12h * 1.1 / 2.0)
    l3_12h = close_12h - (range_12h * 1.1 / 2.0)
    
    # Align 12h Camarilla levels to 4h (wait for completed 12h bar)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    
    # 12h volume confirmation: volume > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume_12h, vol_ma_20, out=np.zeros_like(volume_12h), where=vol_ma_20!=0)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    # 4h choppiness index regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate True Range and ATR for CHOP
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    chop = np.where(
        range_14 > 0,
        100 * np.log10(sum_atr_14 / range_14) / np.log10(14),
        50  # neutral when range is zero
    )
    
    # Regime filter: CHOP > 61.8 = ranging (favor mean reversion), CHOP < 38.2 = trending
    chop_regime = chop > 61.8  # True when ranging/market good for fade
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(pp_12h_aligned[i]) or np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or
            np.isnan(vol_confirm_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price returns to pivot or breaks below L3 (failed breakout)
            if close[i] <= pp_12h_aligned[i] or close[i] < l3_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price returns to pivot or breaks above H3 (failed breakout)
            if close[i] >= pp_12h_aligned[i] or close[i] > h3_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation and ranging regime
            if (close[i] > h3_12h_aligned[i] and  # Break above H3 (resistance)
                vol_confirm_aligned[i] > 1.5 and   # Volume confirmation
                chop_regime[i]):                   # Ranging market (fade the breakout)
                position = 1
                signals[i] = 0.25
            elif (close[i] < l3_12h_aligned[i] and  # Break below L3 (support)
                  vol_confirm_aligned[i] > 1.5 and   # Volume confirmation
                  chop_regime[i]):                   # Ranging market (fade the breakout)
                position = -1
                signals[i] = -0.25
    
    return signals