#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot (S1/R1) breakout with 1d EMA34 trend filter and volume spike
    # Camarilla levels provide high-probability support/resistance. Breakouts with volume
    # confirm institutional participation. 1d EMA34 ensures alignment with daily trend.
    # This combination filters false breakouts and works in both bull and bear markets.
    # Target: 20-50 trades/year to minimize fee drag.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla pivot calculation (based on previous 4h candle)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (S1, R1) for each 4h bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation (stricter)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume + price above 1d EMA34 (uptrend)
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume + price below 1d EMA34 (downtrend)
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Camarilla level or trend reversal vs 1d EMA34
            if position == 1:
                if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0