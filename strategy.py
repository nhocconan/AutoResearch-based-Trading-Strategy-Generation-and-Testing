#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels (R1, S1) with breakout confirmation
# and volume spike filter. In ranging markets (CHOP > 61.8), fade at R1/S1 for mean reversion.
# In trending markets (CHOP < 38.2), breakout continuation at R1/S1 with volume confirmation.
# Uses 1w EMA200 for regime filter to avoid counter-trend trades in strong bear/bull markets.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivots and chop regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data for EMA200 regime filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Calculate 1d Camarilla Pivot Levels (R1, S1, R2, S2, R3, S3, R4, S4) ===
    # Pivot point = (High + Low + Close) / 3
    # Range = High - Low
    # R1 = Close + 1.1 * Range * 1/12
    # S1 = Close - 1.1 * Range * 1/12
    # R2 = Close + 1.1 * Range * 2/12
    # S2 = Close - 1.1 * Range * 2/12
    # R3 = Close + 1.1 * Range * 3/12
    # S3 = Close - 1.1 * Range * 3/12
    # R4 = Close + 1.1 * Range * 4/12
    # S4 = Close - 1.1 * Range * 4/12
    
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r1_1d = close_1d + 1.1 * range_1d * (1.0/12.0)
    s1_1d = close_1d - 1.1 * range_1d * (1.0/12.0)
    r2_1d = close_1d + 1.1 * range_1d * (2.0/12.0)
    s2_1d = close_1d - 1.1 * range_1d * (2.0/12.0)
    r3_1d = close_1d + 1.1 * range_1d * (3.0/12.0)
    s3_1d = close_1d - 1.1 * range_1d * (3.0/12.0)
    r4_1d = close_1d + 1.1 * range_1d * (4.0/12.0)
    s4_1d = close_1d - 1.1 * range_1d * (4.0/12.0)
    
    # === Align 1d Camarilla levels to 6h timeframe ===
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === Calculate 1d Choppiness Index (CHOP) for regime detection ===
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # where ATR1 = True Range, n = 14 period
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_14 / (14 * np.log10(14))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 1w EMA200 for regime filter (avoid counter-trend in strong trends) ===
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 6h Volume Confirmation (20-period average) ===
    # We need to get 6h volume data - resample from 1d is not allowed, so we use get_htf_data for 6h
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup - need enough data for all indicators
    warmup = 200
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        chop_val = chop_aligned[i]
        ema_200 = ema_200_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2x average volume for confirmation
        
        # === REGIME FILTER: Avoid counter-trend trades in strong 1w trends ===
        # Only trade long if price > 1w EMA200 (bull regime)
        # Only trade short if price < 1w EMA200 (bear regime)
        regime_long_allowed = price > ema_200
        regime_short_allowed = price < ema_200
        
        # === STOPLOSS: Fixed ATR-based stop (using 1d ATR) ===
        # Calculate 1d ATR for stoploss
        df_1d_for_atr = get_htf_data(prices, '1d')
        high_1d_atr = df_1d_for_atr['high'].values
        low_1d_atr = df_1d_for_atr['low'].values
        close_1d_atr = df_1d_for_atr['close'].values
        tr1_atr = high_1d_atr - low_1d_atr
        tr2_atr = np.abs(high_1d_atr - np.roll(close_1d_atr, 1))
        tr3_atr = np.abs(low_1d_atr - np.roll(close_1d_atr, 1))
        tr_atr = np.maximum(tr1_atr, np.maximum(tr2_atr, tr3_atr))
        tr_atr[0] = high_1d_atr[0] - low_1d_atr[0]
        atr_1d = pd.Series(tr_atr).ewm(span=10, adjust=False, min_periods=10).mean().values
        atr_aligned_for_sl = align_htf_to_ltf(prices, df_1d_for_atr, atr_1d)
        atr_val = atr_aligned_for_sl[i]
        
        # Check stoploss
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches R2 (take profit) or breaks below S1 (stop)
            if price >= r2 or price < s1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # Short position
            # Exit when price reaches S2 (take profit) or breaks above R1 (stop)
            if price <= s2 or price > r1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine market regime based on Choppiness Index
            # CHOP > 61.8 = ranging market (mean revert at R1/S1)
            # CHOP < 38.2 = trending market (breakout continuation at R1/S1)
            if chop_val > 61.8:  # Ranging market - mean reversion
                # Long when price breaks below S1 with volume (expect reversion to pivot)
                if price < s1 and vol_confirm and regime_long_allowed:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Short when price breaks above R1 with volume (expect reversion to pivot)
                elif price > r1 and vol_confirm and regime_short_allowed:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
            elif chop_val < 38.2:  # Trending market - breakout continuation
                # Long when price breaks above R1 with volume (continuation)
                if price > r1 and vol_confirm and regime_long_allowed:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Short when price breaks below S1 with volume (continuation)
                elif price < s1 and vol_confirm and regime_short_allowed:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
            # In choppy transition zone (38.2 <= CHOP <= 61.8), no new entries
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R1S1_VolumeSpike_ChopRegime"
timeframe = "6h"
leverage = 1.0