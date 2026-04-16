#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop filter.
# Long when price breaks above 12h Donchian(20) upper band AND 1d volume > 1.5x 20-day average AND 1d chop > 61.8 (range regime).
# Short when price breaks below 12h Donchian(20) lower band AND 1d volume > 1.5x 20-day average AND 1d chop > 61.8.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Donchian break.
# Uses discrete position size 0.25. Targets mean reversion in ranging markets with volume confirmation.
# Expected trades: 50-150 total over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev = np.roll(donchian_lower, 1)
    donchian_lower_prev[0] = np.nan
    
    # === 1d Indicators: Volume Spike and Choppiness Index ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Choppiness Index (14-period)
    hl_range = df_1d['high'].values - df_1d['low'].values
    atr_1d = pd.Series(hl_range).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(sum_atr_1d) * np.log10(14) / np.log10((max_high_1d - min_low_1d))
    chop_1d = 100 * chop_denom
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned > 61.8  # Range regime
    
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
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 14 for ATR/chop)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_upper_prev[i]) or np.isnan(donchian_lower_prev[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) or 
            np.isnan(atr_12h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_regime = chop_filter[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band (mean reversion failed)
            if price < donchian_lower[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band (mean reversion failed)
            if price > donchian_upper[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND volume spike AND chop > 61.8 (range)
            if (price > donchian_upper[i] and 
                price <= donchian_upper_prev[i] and  # Ensure break above previous bar's upper band
                vol_spike and chop_regime):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower band AND volume spike AND chop > 61.8 (range)
            elif (price < donchian_lower[i] and 
                  price >= donchian_lower_prev[i] and  # Ensure break below previous bar's lower band
                  vol_spike and chop_regime):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0