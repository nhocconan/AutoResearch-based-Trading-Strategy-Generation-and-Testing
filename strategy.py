#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme reversal with 1w volume spike filter and ATR-based stoploss
# Uses 1d primary timeframe with 1w HTF for volume confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) signal potential reversals.
# Volume spike confirms institutional participation at reversal points.
# ATR stoploss manages risk during trending markets.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while maintaining statistical significance.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data (HTF for volume confirmation) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === 1d Williams %R (14-period) ===
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Align Williams %R to 1d timeframe (no additional delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 1w Volume confirmation (20-period MA spike) ===
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * vol_ma_20_1w)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # === 1d ATR (14-period) for stoploss ===
    atr_1d = np.maximum(
        np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))),
        np.abs(low_1d - np.roll(close_1d, 1))
    )
    atr_1d[0] = high_1d[0] - low_1d[0]  # Fix first value
    atr_ma_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_14)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        vol_conf = vol_spike_aligned[i]
        atr_val = atr_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R returns from oversold (> -50) or shows weakness
            if wr > -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns from overbought (< -50) or shows strength
            if wr < -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike for confirmation
            if vol_conf:
                # Go long when Williams %R is deeply oversold (< -80)
                if wr < -80:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when Williams %R is deeply overbought (> -20)
                elif wr > -20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WilliamsR_Extreme_1wVolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0