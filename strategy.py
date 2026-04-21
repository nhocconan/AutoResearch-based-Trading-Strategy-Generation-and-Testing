#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_Regime_v1
Hypothesis: Camarilla pivot breakout on 6h with 12h EMA trend filter, volume spike confirmation, and choppy market regime filter. Designed for low trade frequency (~15-30/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 6h primary timeframe with 12h HTF for trend and volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # === 12h trend filter: 34-period EMA ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h volume average (20-period) for spike detection ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h[np.isnan(vol_ma_12h)] = 1.0  # avoid division by zero
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === Chop regime filter: Bollinger Band Width percentile on 1d ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width[np.isnan(bb_width)] = 0.1
    
    # Chop regime: BB Width percentile rank (lower = choppy, higher = trending)
    # We want trending markets: BB Width > 60th percentile
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile[np.isnan(bb_width_percentile)] = 0.5
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # === Calculate Camarilla levels from previous 12h bar ===
    # Camarilla levels based on previous bar's range
    close_prev = df_12h['close'].shift(1).values
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    range_prev = high_prev - low_prev
    
    # Camarilla R1, S1, R3, S3, R4, S4
    r1 = close_prev + range_prev * 1.1 / 12
    s1 = close_prev - range_prev * 1.1 / 12
    r3 = close_prev + range_prev * 1.1 / 4
    s3 = close_prev - range_prev * 1.1 / 4
    r4 = close_prev + range_prev * 1.1 / 2
    s4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_12h = ema_34_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        chop_regime = bb_width_percentile_aligned[i]
        
        # Camarilla levels from previous 12h bar
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # Regime filter: only trade in trending markets (BB Width > 60th percentile)
        if chop_regime < 0.6:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 with volume spike and above 12h EMA34
            if price_close > r4 and vol_spike > 2.0 and price_close > trend_12h:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume spike and below 12h EMA34
            elif price_close < s4 and vol_spike > 2.0 and price_close < trend_12h:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit logic: fade at R3/S3 or reverse at R4/S4
            if position == 1:  # Long position
                # Take profit at R3 (fade level)
                if price_high >= r3:
                    signals[i] = 0.0
                    position = 0
                # Stop loss if price breaks below S1 (failed breakout)
                elif price_low <= s1:
                    signals[i] = 0.0
                    position = 0
                # Reverse if price breaks above R4 with strong volume (continuation)
                elif price_close > r4 and vol_spike > 2.5:
                    signals[i] = 0.25  # stay long
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Take profit at S3 (fade level)
                if price_low <= s3:
                    signals[i] = 0.0
                    position = 0
                # Stop loss if price breaks above R1 (failed breakout)
                elif price_high >= r1:
                    signals[i] = 0.0
                    position = 0
                # Reverse if price breaks below S4 with strong volume (continuation)
                elif price_close < s4 and vol_spike > 2.5:
                    signals[i] = -0.25  # stay short
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_Regime_v1"
timeframe = "6h"
leverage = 1.0