#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter, volume spike, and choppiness regime filter.
- Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
- Long when price breaks above R1 with volume spike, 12h uptrend (EMA50), and choppy market (CHOP > 61.8)
- Short when price breaks below S1 with volume spike, 12h downtrend (EMA50), and choppy market (CHOP > 61.8)
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Choppiness filter avoids trending markets where breakouts fail, focuses on ranging markets for mean reversion
- Volume spike confirms institutional participation
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 12h data ONCE before loop for EMA50 and choppiness
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Choppiness Index on 12h for regime filter
    # CHOP = 100 * log10(sum(ATR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_12h[0] - low_12h[0]  # First TR
    
    # ATR(14)
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop_denominator = hh_12h - ll_12h
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_12h = np.where(np.isnan(chop_raw), 50.0, chop_raw)  # Fill NaN with neutral value
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA50, 14+14 for CHOP, 20 for volume MA)
    start_idx = max(50, 14 + 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(chop_12h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation and regime filters
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 12h trend filter
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        # Choppiness filter: only trade when market is choppy (CHOP > 61.8 = ranging)
        choppy_market = chop_12h_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 12h uptrend AND choppy market
            if price_above_R1 and volume_spike[i] and trend_up and choppy_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND 12h downtrend AND choppy market
            elif price_below_S1 and volume_spike[i] and trend_down and choppy_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 12h trend turns down OR market stops being choppy
            if price_below_S1 or not trend_down or not choppy_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 12h trend turns up OR market stops being choppy
            if price_above_R1 or not trend_up or not choppy_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0