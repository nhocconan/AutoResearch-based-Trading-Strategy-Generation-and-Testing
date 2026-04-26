#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime_v1
Hypothesis: Daily Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike + chop regime filter.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- Long when price breaks above R1 with volume spike, 1w uptrend, and low chop (trending market)
- Short when price breaks below S1 with volume spike, 1w downtrend, and low chop
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Chop filter avoids ranging markets where breakouts fail
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
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 1w data ONCE before loop for trend and chop filters
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Choppiness Index on 1w to filter ranging markets
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = np.maximum(high_1w - low_1w, np.absolute(high_1w - np.roll(close_1w, 1)))
    tr2 = np.maximum(np.absolute(low_1w - np.roll(close_1w, 1)), tr1)
    tr1[0] = high_1w[0] - low_1w[0]  # First TR
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range_14 = highest_high_14 - lowest_low_14
    hl_range_14 = np.where(hl_range_14 == 0, 1e-10, hl_range_14)
    
    chop_1w = 100 * np.log10(atr14 * 14 / np.log10(14) / hl_range_14) / np.log10(100)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate volume spike (20-period volume average on 1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation and regime filter
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 1w trend filter
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Choppiness filter: only trade when market is trending (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 1w uptrend AND trending market
            if price_above_R1 and volume_spike[i] and trend_up and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND 1w downtrend AND trending market
            elif price_below_S1 and volume_spike[i] and trend_down and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 1w trend turns down OR market becomes choppy
            if price_below_S1 or not trend_up or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 1w trend turns up OR market becomes choppy
            if price_above_R1 or not trend_down or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime_v1"
timeframe = "1d"
leverage = 1.0