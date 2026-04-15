#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation
# Long when price breaks above R1 with volume > 1.5x 20-period avg and close > R1
# Short when price breaks below S1 with volume > 1.5x 20-period avg and close < S1
# Exit when price returns to the pivot point (PP) level
# Uses 1w trend filter: only take longs when price > 1w EMA50, shorts when price < 1w EMA50
# Designed for low trade frequency (15-25/year) to minimize fee drag in 6h timeframe
# Camarilla levels provide mathematically derived support/resistance that work in ranging markets
# Volume confirmation filters breakouts, 1w EMA filter avoids counter-trend whipsaws

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (based on previous day) ===
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    # R4 = PP + (H - L) * 1.1 / 2
    # S4 = PP - (H - L) * 1.1 / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data (shifted by 1 to avoid look-ahead)
    pp = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3.0
    r1 = pp + (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 12.0
    s1 = pp - (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 12.0
    r4 = pp + (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 2.0
    s4 = pp - (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe (already delayed by roll for previous day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1w Indicator: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50(1w) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above R1 with volume confirmation
        # 2. 1w EMA50 uptrend (price > EMA50) for bias
        # 3. Not already in extreme overbought territory (below R4)
        if (close[i] > r1_aligned[i]) and vol_confirm and \
           (close[i] > ema_50_1w_aligned[i]) and (close[i] < r4_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below S1 with volume confirmation
        # 2. 1w EMA50 downtrend (price < EMA50) for bias
        # 3. Not already in extreme oversold territory (above S4)
        elif (close[i] < s1_aligned[i]) and vol_confirm and \
             (close[i] < ema_50_1w_aligned[i]) and (close[i] > s4_aligned[i]):
            signals[i] = -0.25
        
        # === EXIT CONDITIONS ===
        # Exit long when price returns to pivot point (PP)
        # Exit short when price returns to pivot point (PP)
        elif (signals[i-1] == 0.25 and close[i] <= pp_aligned[i]) or \
             (signals[i-1] == -0.25 and close[i] >= pp_aligned[i]):
            signals[i] = 0.0
        
        else:
            # Hold previous signal
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Camarilla_R1S1_1wEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0