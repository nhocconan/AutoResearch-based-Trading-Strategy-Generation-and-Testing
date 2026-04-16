#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike (1.8x median) and 1d chop regime filter (CHOP > 61.8 = range)
# Long when price > Donchian upper(20) AND 12h volume > 1.8x 30-period median AND 1d CHOP > 61.8
# Short when price < Donchian lower(20) AND 12h volume > 1.8x 30-period median AND 1d CHOP > 61.8
# Exit when price crosses Donchian midpoint (mean reversion to equilibrium)
# Uses discrete position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
# Combines price channel breakout with volume confirmation and range regime filter for robustness in bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Choppiness Index (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d[0] = np.abs(high_1d[0] - low_1d[0])  # first bar
    
    # ATR(14)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR)/ (max(HH)-min(LL))) / log10(14)
    sum_atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = 100 * np.log10(sum_atr_14 / (range_14 + 1e-10)) / np.log10(14)
    chop_1d = np.where(range_14 > 0, chop_1d, 50.0)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Volume median (30-period) ===
    volume_12h = df_12h['volume'].values
    vol_median_30_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).median().values
    vol_median_30_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_30_12h)
    
    # Get 4h data for Donchian levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper/lower
    donch_upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_upper_20 + donch_lower_20) / 2.0
    
    # Align Donchian levels to primary timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_20)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14, 30)  # 1d CHOP, 4h Donchian, 12h volume
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_median_30_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.8x 30-period 12h volume median
        vol_threshold = vol_median_30_12h_aligned[i] * 1.8
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Regime filter: 1d CHOP > 61.8 (range-bound market)
        regime_filter = chop_1d_aligned[i] > 61.8
        
        # Price levels
        price = close[i]
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        mid = donch_mid_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Donchian midpoint (mean reversion)
            if price < mid:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Donchian midpoint (mean reversion)
            if price > mid:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Donchian upper AND volume confirmation AND range regime
            if price > upper and vol_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower AND volume confirmation AND range regime
            elif price < lower and vol_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_12hVolume1.8x_1dChop61.8_v1"
timeframe = "4h"
leverage = 1.0