#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ChopRegime_v1
Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter, volume spike confirmation, and chop regime filter. Designed for low trade frequency (~30-60/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 4h primary timeframe with 12h HTF for trend/volume and 1h for chop regime.
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
    df_1h = get_htf_data(prices, '1h')
    if len(df_12h) < 60 or len(df_1h) < 60:
        return np.zeros(n)
    
    # === 12h trend filter: 50-period EMA ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h volume average (20-period) for spike detection ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h[np.isnan(vol_ma_12h)] = 1.0  # avoid division by zero
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === Chop regime filter (1h timeframe) ===
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    highest_14_1h = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_14_1h = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    
    chop_denom = highest_14_1h - lowest_14_1h
    chop_denom[chop_denom == 0] = 1e-10
    chop = 100 * np.log10(sum(tr[-14:]) / chop_denom[-14:]) / np.log10(14) if False else np.zeros_like(close_1h)  # placeholder
    
    # Proper chop calculation: CHOP = 100 * LOG10(SUM(ATR1, n) / (LOG10(n) * (MAX(HIGH,n) - MIN(LOW,n))))
    # We'll compute it correctly using pandas rolling
    close_1h_series = pd.Series(close_1h)
    high_1h_series = pd.Series(high_1h)
    low_1h_series = pd.Series(low_1h)
    
    tr_1h = pd.Series(np.maximum(high_1h_series - low_1h_series,
                                 np.maximum(np.abs(high_1h_series - close_1h_series.shift(1)),
                                            np.abs(low_1h_series - close_1h_series.shift(1)))))
    tr_1h.iloc[0] = high_1h_series.iloc[0] - low_1h_series.iloc[0]
    
    atr_sum = tr_1h.rolling(window=14, min_periods=14).sum()
    highest_high = high_1h_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_1h_series.rolling(window=14, min_periods=14).min()
    range_hl = highest_high - lowest_low
    range_hl[range_hl == 0] = 1e-10
    
    chop = 100 * np.log10(atr_sum) / (np.log10(14) * np.log10(range_hl))
    chop = chop.values
    chop[np.isnan(chop)] = 50.0  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1h, chop)
    
    # === Camarilla levels from 1d (more stable than 4h) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2, R3, S3, R4, S4
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # R2 = Close + (High - Low) * 1.1/6
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # S2 = Close - (High - Low) * 1.1/6
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    
    camarilla_base = (high_1d - low_1d) * 1.1
    R1 = close_1d + camarilla_base / 12
    S1 = close_1d - camarilla_base / 12
    R2 = close_1d + camarilla_base / 6
    S2 = close_1d - camarilla_base / 6
    R3 = close_1d + camarilla_base / 4
    S3 = close_1d - camarilla_base / 4
    R4 = close_1d + camarilla_base / 2
    S4 = close_1d - camarilla_base / 2
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # === ATR for dynamic stoploss (14-period on 4h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_12h = ema_50_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_14[i]
        
        # Regime filter: chop > 50 = ranging market (good for mean reversion at S1/R1)
        # chop < 50 = trending market (good for breakouts at R2/S2 or R3/S3)
        in_range = chop_val > 50
        in_trend = chop_val <= 50
        
        if position == 0:
            # Long conditions
            if in_range:
                # In ranging market: look for mean reversion at S1
                if price_close <= S1_aligned[i] and vol_spike > 2.0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price_close
                    highest_since_entry = price_close
            else:
                # In trending market: look for breakout at R2
                if price_close >= R2_aligned[i] and vol_spike > 2.0 and price_close > trend_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price_close
                    highest_since_entry = price_close
            
            # Short conditions
            if in_range:
                # In ranging market: look for mean reversion at R1
                if price_close >= R1_aligned[i] and vol_spike > 2.0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price_close
                    lowest_since_entry = price_close
            else:
                # In trending market: look for breakdown at S2
                if price_close <= S2_aligned[i] and vol_spike > 2.0 and price_close < trend_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price_close
                    lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.5 * ATR below highest since entry
                if price_close < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0