#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeVolume_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter (price > 1d EMA34) and volume spike confirmation.
Only trade breakouts in direction of 1d trend to avoid whipsaws. Uses chop regime filter to avoid ranging markets.
Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
Designed for BTC/ETH - Camarilla pivots work in both bull/bear markets via trend/regime filters.
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
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla levels on 1d (based on previous day's OHLC)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = df_1d['high'] - df_1d['low']
    r1 = df_1d['close'] + camarilla_range * 1.1 / 12
    s1 = df_1d['close'] - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Calculate Choppiness Index on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    atr_period = 14
    tr1 = np.maximum(df_1d['high'] - df_1d['low'],
                     np.maximum(abs(df_1d['high'] - df_1d['close'].shift(1)),
                                abs(df_1d['low'] - df_1d['close'].shift(1))))
    atr1 = pd.Series(tr1).rolling(window=atr_period, min_periods=atr_period).mean()
    high_low_14 = pd.Series(df_1d['high']).rolling(window=atr_period, min_periods=atr_period).max() - \
                  pd.Series(df_1d['low']).rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr1 * atr_period) / np.log10(high_low_14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values, additional_delay_bars=0)
    
    # Chop regime: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # We'll use CHOP < 50 as a simpler regime filter (avoid extreme chop)
    chop_regime = chop_aligned < 50  # True when market is not excessively choppy
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 14 for CHOP)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Price breakout conditions
        breakout_above_r1 = close[i] > r1_aligned[i]
        breakdown_below_s1 = close[i] < s1_aligned[i]
        
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long signal: breakout above R1 with volume spike and favorable regime
            if breakout_above_r1 and volume_spike and chop_regime[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: breakdown below S1 OR loss of regime
            elif breakdown_below_s1 or not chop_regime[i]:
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
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short signal: breakdown below S1 with volume spike and favorable regime
            if breakdown_below_s1 and volume_spike and chop_regime[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: breakout above R1 OR loss of regime
            elif breakout_above_r1 or not chop_regime[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeVolume_v1"
timeframe = "12h"
leverage = 1.0