#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 2.0x 20-period average AND CHOP(14) < 40 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 2.0x 20-period average AND CHOP(14) < 40 (trending).
# Exit on opposite Camarilla break (R1 for shorts, S1 for longs) or ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong intraday moves with volume confirmation in trending regimes.
# Works in both bull and bear markets by requiring volume confirmation and choppiness regime filter, avoiding false breakouts in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (R1, S1) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    camarilla_r1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1d Indicators: Choppiness Regime (CHOP(14) < 40 = trending) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14)/(n*ATR)) / log10(n)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_14 / (14 * atr_1d)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned < 40.0  # Trending regime
    
    # === 12h ATR for stoploss ===
    tr1_12h = pd.Series(high_12h).diff()
    tr2_12h = pd.Series(low_12h).diff().abs()
    tr3_12h = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR/CHOP)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_12h_aligned[i]) or np.isnan(camarilla_s1_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) or np.isnan(atr_12h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_regime = chop_filter[i]
        atr_val = atr_12h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1_12h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1_12h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND chop filter (trending)
            if price > camarilla_r1_12h_aligned[i] and vol_spike and chop_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND chop filter (trending)
            elif price < camarilla_s1_12h_aligned[i] and vol_spike and chop_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dVolSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0