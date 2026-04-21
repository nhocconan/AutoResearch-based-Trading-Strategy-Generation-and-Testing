#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: 12h Camarilla pivot (R1/S1) breakout with volume confirmation and choppiness regime filter.
Long when price breaks above R1 with volume spike and chop < 61.8 (trending).
Short when price breaks below S1 with volume spike and chop < 61.8.
Uses 1d HTF EMA50 trend filter to avoid counter-trend trades.
ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR.
Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
Works in bull/bear markets: Camarilla levels provide structure, volume confirms breakouts,
chop filter avoids false signals in ranging markets, HTF trend filter aligns with higher timeframe bias.
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
    
    # === 12h indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use the previous completed 12h bar's OHLC
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    # Avoid look-ahead: only use values from previous bar
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * vol_ma)
    
    # Choppiness regime filter: CHOP < 61.8 = trending (good for breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_period = 14
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high_12h).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero and log of zero/negative
    hh_ll = highest_high - lowest_low
    sum_atr = tr.rolling(window=atr_period, min_periods=atr_period).sum().values
    
    chop = np.zeros_like(close_12h)
    chop[:] = np.nan
    valid = (hh_ll > 0) & (~np.isnan(sum_atr)) & (sum_atr > 0)
    chop[valid] = 100 * np.log10(sum_atr[valid] / hh_ll[valid]) / np.log10(atr_period)
    
    # Chop < 61.8 = trending (favorable for breakouts)
    chop_trending = chop < 61.8
    
    # ATR for stoploss (use same ATR as above)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # start after warmup period
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_spike[i]) 
            or np.isnan(chop_trending[i]) or np.isnan(ema_50_1d_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + trending chop + long HTF bias
            if price > r1[i] and volume_spike[i] and chop_trending[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume spike + trending chop + short HTF bias
            elif price < s1[i] and volume_spike[i] and chop_trending[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below R1 or chop becomes too high (ranging)
            elif price < r1[i] or chop_trending[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above S1 or chop becomes too high (ranging)
            elif price > s1[i] or chop_trending[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "12h"
leverage = 1.0