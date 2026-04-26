#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Regime_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and chop regime filter.
Only trade breakouts in direction of weekly trend when market is not too choppy (CHOP < 61.8).
Volume confirmation ensures momentum. Uses discrete sizing (0.25) to minimize fee churn.
Target: 50-150 total trades over 4 years (12-37/year) by requiring Camarilla breakout,
weekly trend alignment, low chop regime, and volume spike.
Designed for BTC/ETH - Camarilla pivots work in both bull/bear markets via trend/chop filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for HTF trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    htf_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Load daily data for Camarilla pivots and Choppiness index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3 and S3 as primary breakout levels
    camarilla_R3 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    camarilla_S3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3.values)
    
    # Calculate Choppiness Index on daily timeframe
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP = 100 * log10(atr_sum / (log10(n) * range_max))
    # We'll use a practical approximation: CHOP = 100 * (atr_14 / (atr_14 + price_change_14))
    # But standard formula requires true range and n-period calculations
    # Using common implementation: CHOP = 100 * log10(sum(TR(14)) / (log10(14) * (HH(14) - LL(14))))
    
    # Calculate True Range components
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR(14) and max/min over 14 periods
    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
    max_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log10(14) * range_14))
    # Avoid division by zero and log of zero
    atr_sum_14 = atr_14.rolling(window=14, min_periods=14).sum()
    chop_raw = 100 * np.log10(atr_sum_14 / (np.log10(14) * range_14))
    # Replace infinite/NaN values with 50 (neutral)
    chop_values = chop_raw.fillna(50).replace([np.inf, -np.inf], 50).values
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 14+14=28 for CHOP, 20 for volume MA)
    start_idx = max(34, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Choppiness regime: only trade when CHOP < 61.8 (not too choppy)
        low_chop_regime = chop_aligned[i] < 61.8
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_R3_aligned[i]
        breakout_short = close[i] < camarilla_S3_aligned[i]
        
        if htf_trend[i] == 1:  # Uptrend on weekly
            # Long signal: breakout above R3 with volume spike and low chop
            if breakout_long and volume_spike and low_chop_regime:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: breakout below S3 OR chop becomes too high
            elif breakout_short or chop_aligned[i] >= 61.8:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on weekly
            # Short signal: breakout below S3 with volume spike and low chop
            if breakout_short and volume_spike and low_chop_regime:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: breakout above R3 OR chop becomes too high
            elif breakout_long or chop_aligned[i] >= 61.8:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0