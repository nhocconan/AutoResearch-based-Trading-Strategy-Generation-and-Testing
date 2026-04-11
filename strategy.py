#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla pivot breakouts with weekly trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (based on 1d range) act as key support/resistance.
# Breakouts above R4 or below S4 with weekly trend alignment and volume >2x average
# capture institutional moves. Weekly trend filter avoids counter-trend trades.
# Designed for low trade frequency (~15-35/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 12-period EMA for 12h trend filter (faster than 50)
    ema_12_12h = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla levels based on previous day's range
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_r4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_s4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation: current volume > 2.0 x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if np.isnan(ema_12_12h[i]) or np.isnan(ema_20_1w_aligned[i]) or \
           np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or \
           np.isnan(vol_avg_24[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume[i] > 2.0 * vol_avg_24[i]
        
        # Trend filters
        weekly_uptrend = ema_12_12h[i] > ema_20_1w_aligned[i]
        weekly_downtrend = ema_12_12h[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions
        breakout_above_r4 = high[i] > camarilla_r4_aligned[i]
        breakdown_below_s4 = low[i] < camarilla_s4_aligned[i]
        
        # Entry conditions
        # Long: Break above R4 AND weekly uptrend AND volume confirmation
        if breakout_above_r4 and weekly_uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Break below S4 AND weekly downtrend AND volume confirmation
        elif breakdown_below_s4 and weekly_downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Camarilla level touch (touch S4 for long, touch R4 for short)
        elif position == 1 and low[i] <= camarilla_s4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] >= camarilla_r4_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals