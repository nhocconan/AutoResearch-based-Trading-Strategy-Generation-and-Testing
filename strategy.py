#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ATR regime filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND ATR(14) > 1.2 * ATR(50) (high volatility regime) AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND ATR(14) > 1.2 * ATR(50) AND volume > 1.5x 20-bar average.
# Exit on opposite Donchian breakout or ATR regime shift (ATR(14) < 0.8 * ATR(50)).
# Uses discrete position size 0.25. Designed for 6h timeframe with 12h HTF for ATR regime.
# Target: 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets by requiring
# high volatility regime (avoids chop) and volume confirmation to filter false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_high_prev = np.roll(donchian_high, 1)
    donchian_high_prev[0] = np.nan
    donchian_low_prev = np.roll(donchian_low, 1)
    donchian_low_prev[0] = np.nan
    
    # === 6h ATR for regime filter ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50_raw = pd.Series(tr_6h).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    
    # === 12h Indicators: ATR regime (using same ATR but aligned from 12h data for HTF perspective) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1_12h = pd.Series(high_12h).diff()
    tr2_12h = pd.Series(low_12h).diff().abs()
    tr3_12h = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_14_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50_12h_raw = pd.Series(tr_12h).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h_raw)
    atr_50_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h_raw)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 60
    
    # Track position state and entry price for reference
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_high_prev[i]) or
            np.isnan(donchian_low_prev[i]) or np.isnan(atr_14_raw[i]) or np.isnan(atr_50_raw[i]) or
            np.isnan(atr_14_12h_aligned[i]) or np.isnan(atr_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_14 = atr_14_raw[i]
        atr_50 = atr_50_raw[i]
        atr_14_12h = atr_14_12h_aligned[i]
        atr_50_12h = atr_50_12h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (breakout failure)
            if price < donchian_low[i]:
                exit_signal = True
            # Exit if ATR regime shifts to low volatility (chop)
            elif atr_14 < 0.8 * atr_50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (breakout failure)
            if price > donchian_high[i]:
                exit_signal = True
            # Exit if ATR regime shifts to low volatility (chop)
            elif atr_14 < 0.8 * atr_50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # High volatility regime filter: ATR(14) > 1.2 * ATR(50) on both 6h and 12h
            vol_regime_6h = atr_14 > 1.2 * atr_50
            vol_regime_12h = atr_14_12h > 1.2 * atr_50_12h
            
            # LONG: Price breaks above Donchian high AND high vol regime AND volume spike
            if (price > donchian_high_prev[i] and vol_regime_6h and vol_regime_12h and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND high vol regime AND volume spike
            elif (price < donchian_low_prev[i] and vol_regime_6h and vol_regime_12h and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_12hATRRegime_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0