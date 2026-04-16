#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR volatility regime filter and volume confirmation
# Uses 6h primary timeframe with 1d HTF for ATR-based volatility regime (low vol = breakout prone).
# Donchian breakout captures momentum with volume confirmation to avoid false breakouts.
# Volatility regime filter avoids trading during high chop/whipsaw periods.
# Target: 50-150 total trades over 4 years (12-37/year) to balance statistical significance and fee drag.
# Works in both bull and bear markets: breakouts occur in all regimes, but filter improves edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for volatility regime and volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 6h Donchian channels (20-period) ===
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 6h timeframe (wait for 6h bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    
    # === 1d ATR for volatility regime ===
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: low volatility environment (ATR below its 50-period MA)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr_1d < atr_ma_50
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    # === 1d Volume confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(low_vol_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_conf = vol_spike_aligned[i]
        low_vol = low_vol_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low (failed breakout)
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high (failed breakout)
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require both volume confirmation AND low volatility regime
            if vol_conf and low_vol:
                # Go long when price breaks above Donchian high
                if price > donch_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian low
                elif price < donch_low_aligned[i]:
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

name = "6h_Donchian20_VolRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0