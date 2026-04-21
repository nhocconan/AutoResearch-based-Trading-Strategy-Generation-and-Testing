#!/usr/bin/env python3
"""
6h_WeeklyDonchian20_Breakout_1dVolumeSpike_Regime_v1
Hypothesis: Use weekly Donchian(20) breakout direction as HTF trend filter (more reliable than EMAs in chop) + 6h price breakout above/below prior 6h bar's high/low with 1d volume spike (>2.0x 20-period average) for entry confirmation. This avoids false breakouts in ranging markets while capturing strong weekly trends. Target 60-120 trades over 4 years (15-30/year) to stay within 6h fee drag limits. Works in bull via breakout continuation and bear via short breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for Donchian trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian(20) for HTF trend regime ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # === 6h close, high, low for breakout detection ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 1d volume confirmation (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_confirmed = volume_1d > (2.0 * vol_ma_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        donchian_high = donchian_high_20_aligned[i]
        donchian_low = donchian_low_20_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly trend: price above Donchian high = uptrend, below low = downtrend
        uptrend = price > donchian_high
        downtrend = price < donchian_low
        
        if position == 0:
            # Long: price breaks above prior 6h high, weekly uptrend, volume confirmed
            long_condition = (price > high[i-1]) and uptrend and vol_conf
            # Short: price breaks below prior 6h low, weekly downtrend, volume confirmed
            short_condition = (price < low[i-1]) and downtrend and vol_conf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below weekly Donchian low)
                elif price < donchian_low:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above weekly Donchian high)
                elif price > donchian_high:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dVolumeSpike_Regime_v1"
timeframe = "6h"
leverage = 1.0