#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume_v1
Hypothesis: Daily price breaking above Camarilla R1 or below S1 from prior day captures institutional breakouts. Combined with weekly EMA34 trend filter (1w EMA34 slope) and volume spike (>1.8x 20-day MA) to reduce false breakouts. Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring strong momentum confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels, 1w for trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels (standard breakout signals)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Weekly EMA34 trend (with extra delay for confirmation) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=1)
    
    # Weekly EMA34 slope (trend direction)
    ema_slope = np.zeros_like(ema_34_1w_aligned)
    ema_slope[1:] = ema_34_1w_aligned[1:] - ema_34_1w_aligned[:-1]
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = vol_ratio[i]
        weekly_trend_up = ema_slope[i] > 0
        weekly_trend_down = ema_slope[i] < 0
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 1.8 + weekly uptrend
            if price_close > r1 and vol_spike > 1.8 and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + volume spike > 1.8 + weekly downtrend
            elif price_close < s1 and vol_spike > 1.8 and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: price reverts to Camarilla H3/L3 levels (mean reversion)
            high_1d_val = high_1d[i] if i < len(high_1d) else high_1d[-1]
            low_1d_val = low_1d[i] if i < len(low_1d) else low_1d[-1]
            camarilla_h3 = close_1d[i] + (high_1d_val - low_1d_val) * 1.1 / 4 if i < len(close_1d) else close_1d[-1] + (high_1d_val - low_1d_val) * 1.1 / 4
            camarilla_l3 = close_1d[i] - (high_1d_val - low_1d_val) * 1.1 / 4 if i < len(close_1d) else close_1d[-1] - (high_1d_val - low_1d_val) * 1.1 / 4
            
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, camarilla_h3))[i] if i < len(close_1d) else align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, close_1d[-1]))[i]
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, camarilla_l3))[i] if i < len(close_1d) else align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, close_1d[-1]))[i]
            
            if position == 1:
                if price_close < camarilla_h3_aligned:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > camarilla_l3_aligned:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0