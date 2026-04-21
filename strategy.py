#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_Regime_v4
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1d EMA trend filter, volume spike confirmation, and chop regime filter reduce false signals and capture sustained moves. Designed for low trade frequency (~12-37/year) to minimize fee drag and work in both bull/bear markets via regime-adaptive filtering.
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
    
    # === 1d trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume spike filter (20-period on 1d) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Choppiness regime filter (14-period on 1d) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === ATR for dynamic stoploss (14-period on 1d) ===
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_1d_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        
        # Calculate Camarilla levels from prior 12h session (using 1d HTF data as proxy)
        # Since we don't have direct 12h data, we approximate using 1d OHLC of previous day
        # This is a simplification; in practice would need 12h data but we use 1d as HTF reference
        if i >= 2:  # Need at least 2 bars for previous period approximation
            # Use previous bar's high/low/close as proxy for prior session
            prev_high = prices['high'].iloc[i-1]
            prev_low = prices['low'].iloc[i-1]
            prev_close = prices['close'].iloc[i-1]
            
            camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
            camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
        else:
            camarilla_r1 = price_close
            camarilla_s1 = price_close
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 1.5 + price above 1d EMA + chop < 61.8 (trending market)
            if price_close > camarilla_r1 and vol_spike > 1.5 and price_close > trend_1d and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below S1 + volume spike > 1.5 + price below 1d EMA + chop < 61.8 (trending market)
            elif price_close < camarilla_s1 and vol_spike > 1.5 and price_close < trend_1d and chop_val < 61.8:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_Regime_v4"
timeframe = "12h"
leverage = 1.0