#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopRegime_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike > 2.0, and chop regime filter (CHOP > 61.8 for range/mean reversion). Designed for low trade frequency (~20-40/year) to minimize fee drag. Uses Camarilla levels from prior day for structure, volume confirmation for conviction, and chop filter to avoid whipsaws in strong trends. Works in bull/bear via mean reversion in ranging markets and trend alignment in trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
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
    
    # === Choppiness Index (14-period on 4h) for regime filter ===
    # CHOP > 61.8 = ranging (good for mean reversion), CHOP < 38.2 = trending
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum_14 / np.log10(14) / (highest_14 - lowest_14))
    chop[np.isnan(chop) | (highest_14 - lowest_14) == 0] = 50.0  # default to neutral when invalid
    
    # === Camarilla levels from prior 1d (using OHLC of completed 1d bar) ===
    # Need prior day's OHLC, so shift by 1 to avoid look-ahead
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today based on yesterday's OHLC
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    camarilla_range = (high_1d - low_1d)
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align to 4h: today's R3/S3 based on yesterday's OHLC, available after 1d close
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(chop[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        chop_val = chop[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike > 2.0 + chop > 61.8 (ranging) + price above 1d EMA34
            if price_close > r3_level and vol_spike > 2.0 and chop_val > 61.8 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 + volume spike > 2.0 + chop > 61.8 (ranging) + price below 1d EMA34
            elif price_close < s3_level and vol_spike > 2.0 and chop_val > 61.8 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Trailing stop: 2.0 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0