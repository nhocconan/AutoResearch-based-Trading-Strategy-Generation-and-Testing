#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v2
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d trend filter and choppiness regime filter.
Only trade breakouts in direction of 1d EMA34 trend when market is not too choppy (CHOP < 61.8).
Avoids whipsaws in ranging markets while capturing trends in both bull and bear markets.
Volume confirmation reduces false breakouts. Discrete position sizing (0.25) minimizes fee churn.
Target: 50-150 total trades over 4 years (12-37/year) by requiring confluence of breakout, trend, volume, and chop filter.
Designed for BTC/ETH - avoids SOL-only bias by requiring HTF trend alignment and regime filter.
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
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])  # R1 level
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])  # S1 level
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Calculate 20-period average true range for choppiness index (14-period)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]]))).values
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]]))).values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR14) / (n * (max(high)-min(low)))) / log10(n)
    # Simplified: CHOP < 61.8 = trending, CHOP > 61.8 = ranging
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade when market is trending (not choppy)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 14 for ATR/CHOP, 20 for volume MA)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        # Breakout conditions with trend filter and chop filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above R1 with volume spike and chop filter (trade with trend)
            if close[i] > R1_1d_aligned[i] and volume_spike and chop_filter[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Mean reversion short: breakdown below S1 in uptrend (fade the move) only in low chop
            elif close[i] < S1_1d_aligned[i] and volume_spike and chop_filter[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short breakdown below S1 with volume spike and chop filter (trade with trend)
            if close[i] < S1_1d_aligned[i] and volume_spike and chop_filter[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Mean reversion long: breakout above R1 in downtrend (fade the move) only in low chop
            elif close[i] > R1_1d_aligned[i] and volume_spike and chop_filter[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v2"
timeframe = "12h"
leverage = 1.0