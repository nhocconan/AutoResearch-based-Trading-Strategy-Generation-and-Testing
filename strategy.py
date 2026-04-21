#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter (EMA34) and 1d volume spike confirmation reduces false breakouts while capturing institutional moves. Chop regime filter (4h) avoids ranging markets. Designed for low trade frequency (~15-35/year) to minimize fee drag and improve test generalization in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === Camarilla levels from prior 1-hour session (HLC of previous 1h bar) ===
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Camarilla R1, S1 levels (breakout signals) based on prior 1h bar
    camarilla_r1 = close_1h + (high_1h - low_1h) * 1.1 / 12
    camarilla_s1 = close_1h - (high_1h - low_1h) * 1.1 / 12
    
    # Note: Camarilla levels are for current bar, no alignment needed
    
    # === 4h trend filter: 34-period EMA on 4h ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d volume spike filter (20-period on 1d) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Choppiness regime filter (14-period on 4h) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(atr_4h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        trend_4h = ema_34_4h_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 4h EMA + chop < 61.8 (trending market)
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_4h and chop_val < 61.8:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below 4h EMA + chop < 61.8 (trending market)
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_4h and chop_val < 61.8:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit on opposite Camarilla level break or volume dry-up
            if position == 1:
                # Exit long if price breaks below S1 or volume drops below average
                if price_close < s1 or vol_spike < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short if price breaks above R1 or volume drops below average
                if price_close > r1 or vol_spike < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike_v1"
timeframe = "1h"
leverage = 1.0