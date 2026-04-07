#!/usr/bin/env python3
"""
6H Heikin-Ashi Trend with Volume Confirmation and 12h ADX Filter
Long when Heikin-Ashi shows bullish candle (close > open) with volume above average AND 12h ADX > 25
Short when Heikin-Ashi shows bearish candle (close < open) with volume above average AND 12h ADX > 25
Exit when Heikin-Ashi candle changes direction
Heikin-Ashi smooths price action to filter noise, while ADX ensures we only trade in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_heikin_ashi_trend_volume_12h_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Heikin-Ashi calculation ===
    ha_close = (open_prices + high + low + close) / 4
    ha_open = np.zeros_like(close)
    ha_open[0] = (open_prices[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
    
    # Bullish: HA close > HA open, Bearish: HA close < HA open
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 12h ADX filter (trend strength) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_12h + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_12h + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        if (np.isnan(ha_open[i]) or np.isnan(ha_close[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Heikin-Ashi turns bearish
            if ha_bearish[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Heikin-Ashi turns bullish
            if ha_bullish[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation and strong trend (ADX > 25)
            if vol_ratio[i] < 1.1 or adx_12h_aligned[i] < 25:
                signals[i] = 0.0
                continue
            
            # Entry: Heikin-Ashi direction with volume and ADX confirmation
            if ha_bullish[i]:
                # Bullish HA candle with volume and trend -> long
                position = 1
                signals[i] = 0.25
            elif ha_bearish[i]:
                # Bearish HA candle with volume and trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals