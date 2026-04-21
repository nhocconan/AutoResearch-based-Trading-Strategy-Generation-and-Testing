#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dATRRegime_VolumeSpike_v1
Hypothesis: Use 12h timeframe with 1d ATR regime filter to avoid choppy markets, combined with Camarilla R1/S1 breakouts and strict volume confirmation (>3.0x 20-period average). Target 50-150 trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear via ATR-based regime filter (low ATR = range, high ATR = trending) that adapts to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for ATR regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d ATR (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # ATR regime: high ATR (> 0.75 * 50-period median) = trending market
    atr_median_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).median().values
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    atr_regime = atr_1d_aligned > (0.75 * atr_median_50_aligned)
    
    # === 12h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # === 12h volume confirmation (volume > 3.0x 20-period average - STRICT) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (3.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_median_50_aligned[i]) or 
            np.isnan(atr_regime[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        regime_ok = atr_1d_aligned[i] > (0.75 * atr_median_50_aligned[i])
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Long: price closes above R1, volume confirmed, ATR regime (trending)
            long_condition = (price > r1_val) and vol_conf and regime_ok
            # Short: price closes below S1, volume confirmed, ATR regime (trending)
            short_condition = (price < s1_val) and vol_conf and regime_ok
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.30 if position == 1 else -0.30
                continue
            
            # Check stoploss (3.0x ATR)
            if position == 1:
                if price < entry_price - 3.0 * atr_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if price > entry_price + 3.0 * atr_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0