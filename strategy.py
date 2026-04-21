#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_Regime_v1
Hypothesis: For 12h timeframe, Camarilla R1/S1 breakouts from prior 12h session with 1d EMA34 trend filter, volume spike confirmation (>2.0), and chop regime (<61.8) captures institutional moves in both bull/bear markets with low trade frequency (~15-35/year). ATR trailing stop (3.0x) manages risk. Designed to avoid overtrading and fee drag while providing edge in ranging and trending regimes.
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === Camarilla levels from prior 12-hour session (HLC of previous 12h bar) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R1, S1 levels (breakout signals)
    camarilla_r1 = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_s1 = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # === 1d trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume spike filter (20-period on 12h) ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === Choppiness regime filter (14-period on 12h) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # === ATR for dynamic stoploss (14-period on 12h) ===
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_14_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_12h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_14_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 1d EMA + chop < 61.8 (trending market)
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_1d and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below S1 + volume spike > 2.0 + price below 1d EMA + chop < 61.8 (trending market)
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_1d and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 3.0 * ATR below highest since entry
                if price_close < highest_since_entry - 3.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 3.0 * ATR above lowest since entry
                if price_close > lowest_since_entry + 3.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_Regime_v1"
timeframe = "12h"
leverage = 1.0