#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: 12h Camarilla pivot R1/S1 breakout with volume confirmation and choppiness regime filter.
Long on break above R1 when choppy market (CHOP > 61.8); Short on break below S1 when choppy market.
Uses 1d HTF EMA50 trend filter to avoid counter-trend trades in strong trends.
ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR.
Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull/bear markets.
Chop regime ensures mean reversion in ranging markets; EMA filter avoids major trend fights.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We need previous bar's high, low, close
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    R1 = pivot + 1.1 * range_hl / 12
    S1 = pivot - 1.1 * range_hl / 12
    
    # Choppiness Index (CHOP) for regime filter - 14 period
    # CHOP = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    # Volume confirmation - volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * vol_ma)
    
    # ATR (14-period) for stoploss
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(chop_values[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Choppiness regime: CHOP > 61.8 indicates ranging market (mean revert)
            in_chop_regime = chop_values[i] > 61.8
            
            # Long: price breaks above R1 + volume spike + chop regime + long HTF bias
            if (price > R1[i] and volume_spike[i] and in_chop_regime 
                and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume spike + chop regime + short HTF bias
            elif (price < S1[i] and volume_spike[i] and in_chop_regime 
                  and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below R1 or chop regime ends (trending)
            elif price < R1[i] or chop_values[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above S1 or chop regime ends (trending)
            elif price > S1[i] or chop_values[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "12h"
leverage = 1.0