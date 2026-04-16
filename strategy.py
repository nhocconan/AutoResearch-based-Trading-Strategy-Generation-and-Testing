#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Uses ATR-based trailing stop (highest high since entry - 2.5*ATR for long, lowest low + 2.5*ATR for short).
# Designed to capture strong intraday moves with volume confirmation in trending regimes, avoiding choppy markets.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on prior day) ===
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use prior 4h bar's high/low/close for today's levels (no look-ahead)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = prior_low[0] = prior_close[0] = np.nan  # first bar has no prior
    camarilla_r1 = prior_close + 1.1 * (prior_high - prior_low) / 12
    camarilla_s1 = prior_close - 1.1 * (prior_high - prior_low) / 12
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h Indicators: Chopiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest(high,14) - lowest(low,14))) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop = 100 * np.log10(atr_sum_14 / chop_denom) / np.log10(14)
    chop_filter = chop < 61.8  # trending regime (CHOP < 61.8)
    
    # === 4h ATR for stoploss calculation ===
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_4h_raw[i]) or np.isnan(chop[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        is_trending = chop_filter[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based trailing stop: highest high since entry - 2.5*ATR
            elif price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1[i]:
                exit_signal = True
            # ATR-based trailing stop: lowest low since entry + 2.5*ATR
            elif price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and is_trending:
            # LONG: Price breaks above Camarilla R1 AND volume spike
            if price > camarilla_r1[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike
            elif price < camarilla_s1[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            # Update trailing stop levels
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = lowest_since_entry  # unchanged for long
            elif position == -1:
                highest_since_entry = highest_since_entry  # unchanged for short
                lowest_since_entry = min(lowest_since_entry, price)
            
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_VolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0