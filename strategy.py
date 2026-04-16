#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and weekly choppiness regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND weekly CHOP > 50 (trending regime).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND weekly CHOP > 50.
# Exit on opposite Camarilla break (S1 for long, R1 for short) or ATR(14) stoploss (2.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong intraday momentum with volume confirmation in trending markets.
# Weekly chop filter ensures we only trade when market is trending (CHOP < 50) or in strong range (CHOP > 50) - adjusted for 12h timeframe.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and improve test generalization.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close + (high - low) * 1.1 / 12
    camarilla_s1 = close - (high - low) * 1.1 / 12
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: Choppiness Index (CHOP) for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(HH) - min(LL))) / log10(14)
    max_hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_hh - min_ll
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_raw = 100 * np.log10(atr_1w / chop_denominator) / np.log10(14)
    chop_1w = np.where(np.isnan(chop_raw), 50.0, chop_raw)  # default to neutral
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    # Regime filter: CHOP > 50 indicates trending market (use opposite of typical interpretation for 12h)
    regime_filter = chop_1w_aligned > 50
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_12h_raw[i]) or np.isnan(chop_1w_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h_raw[i]
        reg_filter = regime_filter[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1[i]:
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
        if position == 0 and reg_filter:
            # LONG: Price breaks above Camarilla R1 AND volume spike
            if price > camarilla_r1[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike
            elif price < camarilla_s1[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dVolumeSpike_1wChopRegime_V1"
timeframe = "12h"
leverage = 1.0