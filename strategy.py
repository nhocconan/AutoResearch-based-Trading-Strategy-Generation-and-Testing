#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1
Hypothesis: Price breaking above Camarilla R1 or below S1 from prior 12h session on the 12h timeframe captures institutional breakouts with lower false signals. Combined with daily trend filter (price above/below 34-period EMA on 1d) and volume spike (>1.5x 20-period MA on 12h). Designed for low trade frequency (~12-30/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring strong momentum confirmation and alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Camarilla levels, 1d for trend and volume MA)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 2 or len(df_1d) < 34:
        return np.zeros(n)
    
    # === Camarilla levels from prior 12-hour session (HLC of previous 12h bar) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R1, S1 levels (breakout signals)
    camarilla_r1 = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_s1 = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # === Daily trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume spike filter (20-period on 12h) ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_12h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        daily_ema = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 1.5 + price above daily EMA (bullish trend)
            if price_close > r1 and vol_spike > 1.5 and price_close > daily_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + volume spike > 1.5 + price below daily EMA (bearish trend)
            elif price_close < s1 and vol_spike > 1.5 and price_close < daily_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR(14) from entry
            # Calculate ATR using 12h data
            if i >= 14:
                high_12h = df_12h['high'].values
                low_12h = df_12h['low'].values
                close_12h = df_12h['close'].values
                
                tr1 = high_12h - low_12h
                tr2 = np.abs(high_12h - np.roll(close_12h, 1))
                tr3 = np.abs(low_12h - np.roll(close_12h, 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr[0] = tr1[0]
                atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
                atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
                
                if position == 1:
                    if price_close < entry_price - 2.5 * atr_14_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # position == -1
                    if price_close > entry_price + 2.5 * atr_14_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # Hold position until ATR is ready
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0