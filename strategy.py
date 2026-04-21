#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v6
Hypothesis: For 12h timeframe, Camarilla R1/S1 breakout with daily EMA trend filter and volume confirmation (using 12h volume vs its 20-period MA) captures institutional breakouts. Uses ATR-based trailing stop for risk management. Designed for low trade frequency (~12-30/year) to minimize fee drag and work in both bull and bear regimes by requiring alignment with higher timeframe trend and momentum confirmation.
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
    # We need to synthesize 12h data from the prices dataframe (which is 12h timeframe)
    # Since we are on 12h timeframe, we can use the previous bar's HLC
    high_12h = prices['high'].shift(1).values
    low_12h = prices['low'].shift(1).values
    close_12h = prices['close'].shift(1).values
    
    # Camarilla R1, S1 levels (breakout signals)
    camarilla_r1 = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_s1 = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Since we are using shift(1), no need for HTF alignment - already aligned to current bar
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    
    # === Daily trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume spike filter (20-period on 12h) ===
    volume_12h = prices['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_12h
    # No alignment needed as volume is already on 12h timeframe
    
    # === ATR for dynamic stoploss (14-period on 12h) ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # No alignment needed as ATR is already on 12h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_12h[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_12h[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        daily_ema = ema_34_1d_aligned[i]
        atr_val = atr_14[i]
        
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v6"
timeframe = "12h"
leverage = 1.0