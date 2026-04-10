#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot long strategy with daily volume spike and choppiness regime filter
# - Long when price touches Camarilla L3 level AND daily volume > 2x 20-day volume SMA
# - Only in choppy markets: Choppiness Index(14) > 61.8 (range-bound regime)
# - Exit: Price reaches Camarilla H3 level or Donchian(20) midpoint
# - Position sizing: 0.25 discrete level
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Combines mean reversion at pivot levels with volume confirmation and regime filter
# - Works in both bull and bear markets by fading extremes in ranging conditions

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels for exit reference
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate Choppiness Index(14) for regime filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # handle division by zero
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    # Camarilla formulas: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)  # Resistance level 3
    l3 = pivot - (range_hl * 1.1 / 4)  # Support level 3
    h4 = pivot + (range_hl * 1.1 / 2)  # Resistance level 4
    l4 = pivot - (range_hl * 1.1 / 2)  # Support level 4
    
    # Align HTF Camarilla levels to LTF (already delayed by get_htf_data + align_htf_to_ltf)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate daily volume spike filter
    df_1d_vol = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(df_1d_vol).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    for i in range(20, n):  # Start after Donchian period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(chop[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Choppiness Index > 61.8 indicates range-bound market
        chop_regime = chop[i] > 61.8
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Daily volume spike: today's volume > 2x 20-day average
        daily_vol_spike = df_1d_vol[i // 24] > 2 * vol_sma_20_1d_aligned[i] if i // 24 < len(df_1d_vol) else False
        
        # Price proximity to Camarilla levels (within 0.1% tolerance)
        tol = 0.001
        near_l3 = abs(close[i] - l3_aligned[i]) / close[i] < tol
        near_h3 = abs(close[i] - h3_aligned[i]) / close[i] < tol
        
        if position == 0:  # Flat - look for long entry at L3
            if near_l3 and chop_regime and vol_confirm and daily_vol_spike:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit at H3 level or Donchian midpoint
            exit_condition = near_h3 or (close[i] > donchian_mid[i])
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
    
    return signals