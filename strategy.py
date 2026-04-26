#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter and volume spike confirmation. 
In bull/bear markets, price often retests prior day's Camarilla levels before continuing trend. 
Breakout above R1 (bullish) or below S1 (bearish) with volume spike and aligned 1d trend captures 
continuation moves. Uses discrete sizing (0.25) to minimize fee churn. Targets 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for stoploss
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d volume average for volume spike filter
    vol_avg_1d = pd.Series(df_1d['volume'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(htf_trend[i]) or np.isnan(vol_avg_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Need prior day's OHLC for Camarilla calculation
        if i < 96:  # Need at least 96 4h bars (4 days) to get prior day's complete OHLC
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get prior day's OHLC (24 hours = 96 4h bars)
        prior_day_idx = i - 96
        if prior_day_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Prior day's OHLC
        phigh = high[prior_day_idx:prior_day_idx+24].max()
        plow = low[prior_day_idx:prior_day_idx+24].min()
        pclose = close[prior_day_idx:prior_day_idx+24].last() if hasattr(np.ndarray, 'last') else close[prior_day_idx+23]
        
        # Calculate Camarilla levels for prior day
        range_val = phigh - plow
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        camarilla_r1 = pclose + range_val * 1.1 / 12
        camarilla_s1 = pclose - range_val * 1.1 / 12
        
        # Volume confirmation: current volume > 1.5x prior day average volume
        vol_spike = volume[i] > (vol_avg_1d_aligned[i] * 1.5)
        
        # Breakout logic
        if htf_trend[i] == 1:  # 1d uptrend
            # Long breakout above R1
            if close[i] > camarilla_r1 and vol_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price breaks below S1 (reversal signal)
            elif position == 1 and close[i] < camarilla_s1:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:  # 1d downtrend
            # Short breakout below S1
            if close[i] < camarilla_s1 and vol_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price breaks above R1 (reversal signal)
            elif position == -1 and close[i] > camarilla_r1:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0