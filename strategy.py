#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v5
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation captures institutional moves with low trade frequency. Uses ATR trailing stop for risk management. Designed for 12h timeframe to target 50-150 total trades over 4 years, minimizing fee drag while working in both bull and bear regimes by requiring alignment with higher timeframe trend and momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === Camarilla levels from prior 12-hour session (HLC of previous 12h bar) ===
    # For 12h timeframe, we need to use 12h data for Camarilla calculation
    # Since mtf_data doesn't have 12h as HTF option, we'll use 4h data and aggregate
    # But actually, we can use the prices DataFrame directly for 12h calculation
    # However, to follow MTF rules properly, let's use 4h as base and compute 12h levels
    # But the instruction says to use mtf_data for HTF, so we'll use 1d for trend and volume
    # and compute Camarilla from 12h price action using the prices DataFrame
    
    # Actually, for 12h timeframe primary, we can compute Camarilla from 12h price data
    # We need to get the previous 12h bar's HLC
    # Since we don't have a direct 12h HTF loader, we'll use the prices DataFrame
    # but shift by 1 bar to get previous completed 12h bar
    
    # Get previous 12h bar's high, low, close (using shift(1) on 12h data)
    # But we need to be careful about look-ahead
    # Instead, let's use 4h data to compute 12h levels properly
    
    # Load 4h data for Camarilla calculation (since 12h = 3*4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # For 12h Camarilla, we need every 3rd 4h bar (completed 12h session)
    # But to avoid complexity and look-ahead, we'll use a simpler approach:
    # Use the current prices DataFrame to get 12h OHLC by grouping
    # However, this violates MTF rules if we resample
    
    # Correct approach: use 4h data and compute Camarilla for each 4h bar
    # but only trade when we have a completed 12h bar (every 3rd 4h bar)
    # But this complicates the logic
    
    # Simpler and correct: since we're on 12h timeframe, each bar in prices is 12h
    # So we can use prices.iloc[i-1] for previous completed 12h bar
    # This is NOT look-ahead because we're using past data
    
    # Actually, looking at the current prices DataFrame, each row IS a 12h bar
    # So we can safely use shift(1) to get previous completed 12h bar
    
    high_12h = prices['high'].shift(1).values
    low_12h = prices['low'].shift(1).values
    close_12h = prices['close'].shift(1).values
    
    # Camarilla R1, S1 levels (breakout signals)
    camarilla_r1 = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_s1 = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # No need to align since we're already on 12h timeframe
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    
    # === Daily trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Daily volume confirmation: volume > 1.5x 20-period EMA on 1d ===
    volume_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ema_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for dynamic stoploss (14-period on 1d) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        daily_ema = ema_34_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 1.5 + price above daily EMA (bullish trend)
            if price_close > r1 and vol_spike > 1.5 and price_close > daily_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below S1 + volume spike > 1.5 + price below daily EMA (bearish trend)
            elif price_close < s1 and vol_spike > 1.5 and price_close < daily_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.5 * ATR below highest since entry
                if price_close < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v5"
timeframe = "12h"
leverage = 1.0