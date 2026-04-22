#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: Daily pivot point (PP) breakout with weekly EMA trend filter and volume confirmation
    # Pivot points provide key support/resistance levels. Breaking above/below with volume 
    # indicates institutional participation. Weekly EMA ensures alignment with long-term trend.
    # This combination filters false breakouts and works in both bull and bear markets.
    # Target: 15-25 trades/year on daily timeframe.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (PP, R1, S1)
    # Pivot Point = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = (2 * pp_1d) - low_1d
    s1_1d = (2 * pp_1d) - high_1d
    
    # Align pivot points to daily timeframe (no alignment needed for same timeframe)
    pp_aligned = pp_1d
    r1_aligned = r1_1d
    s1_aligned = s1_1d
    
    # Load weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(34, n):  # Start after weekly EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume + price above weekly EMA34 (uptrend)
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume + price below weekly EMA34 (downtrend)
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to pivot point or trend reversal vs weekly EMA34
            if position == 1:
                if close[i] < pp_aligned[i] or close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pp_aligned[i] or close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_PivotPoint_R1S1_Breakout_WeeklyEMA34_Trend_Volume_Session_v1"
timeframe = "1d"
leverage = 1.0