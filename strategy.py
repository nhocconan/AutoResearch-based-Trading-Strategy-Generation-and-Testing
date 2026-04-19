#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with daily volume spike and weekly trend filter.
# Long at S1/S2 when price reverses up with volume spike and weekly trend up.
# Short at R1/R2 when price reverses down with volume spike and weekly trend down.
# Uses daily Camarilla levels for intraday mean reversion, weekly trend for direction filter.
# Volume spike confirms momentum behind reversal. Designed to work in ranging markets (2022-2024)
# and capture reversals in bear market rallies/pullbacks (2025+). Target: 20-35 trades/year per symbol.
name = "6h_CamarillaReversal_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily OHLC
    # Using previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla multipliers
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    R2 = prev_close + (prev_high - prev_low) * 1.1666
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    S2 = prev_close - (prev_high - prev_low) * 1.1666
    
    # Align daily Camarilla levels to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_6h = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume confirmation: current volume > 2.0x 24-period average (48 hours)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure volume MA and shifted data are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_6h[i]) or np.isnan(R2_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(S2_6h[i]) or np.isnan(weekly_ema50_6h[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_6h[i]
        r2 = R2_6h[i]
        s1 = S1_6h[i]
        s2 = S2_6h[i]
        weekly_ema = weekly_ema50_6h[i]
        vol_ma = vol_ma_24[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        # Weekly trend filter
        weekly_uptrend = price > weekly_ema
        weekly_downtrend = price < weekly_ema
        
        if position == 0:
            # Enter long at S1/S2 reversal with volume and weekly uptrend
            if price <= s1 and weekly_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short at R1/R2 reversal with volume and weekly downtrend
            elif price >= r1 and weekly_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price reaches midpoint (neutral) or weekly trend turns down
            midpoint = (s1 + r1) / 2
            if price >= midpoint or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price reaches midpoint or weekly trend turns up
            midpoint = (s1 + r1) / 2
            if price <= midpoint or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals