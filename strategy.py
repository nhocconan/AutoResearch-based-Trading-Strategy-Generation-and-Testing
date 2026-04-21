#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_ChopRegime_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter. Designed for low trade frequency (~30-50/year) to minimize fee drag. Uses 4h primary timeframe with 1d HTF for trend, volume, and chop context. Works in bull/bear via regime-adaptive logic: trend follow in trending markets (CHOP<38.2), mean revert in ranging markets (CHOP>61.8).
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
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d choppiness index (14-period) for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d_arr, 1)),
                                  np.abs(low_1d - np.roll(close_1d_arr, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_denom = np.log10(atr_14_1d * 14) / np.log10(2)
    chop_num = highest_high_1d - lowest_low_1d
    chop_num[chop_num <= 0] = 1e-10
    chop_denom[chop_denom <= 0] = 1e-10
    chop_1d = 100 * np.log10(chop_num / chop_denom)
    chop_1d[np.isnan(chop_1d)] = 50.0  # neutral when undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
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
    
    # === Camarilla pivot levels from previous 1d (for 4h breakout) ===
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_multiplier = 1.1 / 12
    camarilla_r1 = close_1d + camarilla_multiplier * (high_1d - low_1d)
    camarilla_s1 = close_1d - camarilla_multiplier * (high_1d - low_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        chop_value = chop_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        atr_val = atr_14[i]
        
        # Regime determination: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        if position == 0:
            # Long conditions
            long_breakout = price_close > r1_level
            long_volume = vol_spike > 2.0
            long_trend_filter = price_close > trend_1d  # above 1d EMA34
            
            # Short conditions
            short_breakout = price_close < s1_level
            short_volume = vol_spike > 2.0
            short_trend_filter = price_close < trend_1d  # below 1d EMA34
            
            if is_trending:
                # Trend following mode: breakout with trend filter
                if long_breakout and long_volume and long_trend_filter:
                    signals[i] = 0.30
                    position = 1
                    entry_price = price_close
                elif short_breakout and short_volume and short_trend_filter:
                    signals[i] = -0.30
                    position = -1
                    entry_price = price_close
            elif is_ranging:
                # Mean reversion mode: fade extreme moves
                if long_breakout and long_volume and price_close < trend_1d:
                    # Fade breakout: short when price above R1 but below trend
                    signals[i] = -0.30
                    position = -1
                    entry_price = price_close
                elif short_breakout and short_volume and price_close > trend_1d:
                    # Fade breakdown: long when price below S1 but above trend
                    signals[i] = 0.30
                    position = 1
                    entry_price = price_close
        
        elif position != 0:
            # ATR-based trailing stop
            if position == 1:
                # Trailing stop: 2.5 * ATR below highest close since entry
                # Simplified: use close-based trailing stop for signal generation
                if price_close < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Trailing stop: 2.5 * ATR above lowest close since entry
                if price_close > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0