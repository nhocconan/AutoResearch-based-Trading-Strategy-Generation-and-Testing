#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla Pivot S1/S2 breakout with 1d EMA34 trend and volume spike
    # Camarilla pivots provide institutional-grade support/resistance levels.
    # Breakouts above S1 or below S2 with volume confirmation indicate strong moves.
    # 1d EMA34 filters for higher timeframe trend direction.
    # Target: 12-37 trades/year (50-150 total over 4 years) with low drawdown.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Camarilla pivots (using daily high/low/close)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Pivot = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    s2_12h = close_12h - (range_12h * 1.1 / 6)
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    r2_12h = close_12h + (range_12h * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(s1_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or
            np.isnan(r1_12h_aligned[i]) or np.isnan(r2_12h_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume + price above 1d EMA34 (uptrend)
            if close[i] > r1_12h_aligned[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume + price below 1d EMA34 (downtrend)
            elif close[i] < s1_12h_aligned[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to pivot or trend reversal vs 1d EMA34
            pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
            if position == 1:
                if close[i] < pivot_12h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_12h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_S1_R1_Breakout_1dEMA34_VolumeSpike_Session_v1"
timeframe = "12h"
leverage = 1.0