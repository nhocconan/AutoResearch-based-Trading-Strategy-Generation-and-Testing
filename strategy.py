#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_Regime_v1
Hypothesis: 4h Camarilla R1/S1 breakout with volume spike and choppiness regime filter. 
In trending markets (CHOP < 38.2), trade breakouts in direction of trend. 
In ranging markets (CHOP > 61.8), fade extreme touches of R1/S1. 
Uses 1d HTF EMA34 for higher timeframe trend alignment. 
Target: 75-200 trades over 4 years by requiring confluence of breakout, volume, regime, and HTF trend. 
Works in bull/bear via adaptive logic: trend-following breakouts in trends, mean reversion at pivots in chop. 
Discrete position sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for Camarilla width and stoploss
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
    
    # Calculate Camarilla levels from previous day
    # Need previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_range[0] = 1e-10  # avoid division by zero
    
    # Camarilla levels
    R4 = prev_close + camarilla_range * 1.5
    R3 = prev_close + camarilla_range * 1.25
    R2 = prev_close + camarilla_range * 1.166
    R1 = prev_close + camarilla_range * 1.083
    S1 = prev_close - camarilla_range * 1.083
    S2 = prev_close - camarilla_range * 1.166
    S3 = prev_close - camarilla_range * 1.25
    S4 = prev_close - camarilla_range * 1.5
    
    # Calculate Choppiness Index (CHOP) for regime filtering
    chop_period = 14
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum()
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max()
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min()
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(chop_period)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Volume spike filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for vol MA, 14 for ATR/CHOP)
    start_idx = max(20, chop_period, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(htf_trend[i]) or
            np.isnan(vol_ma.iloc[i] if hasattr(vol_ma, 'iloc') else vol_ma[i]) if i < len(vol_ma) else True):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_spike_i = vol_spike.iloc[i] if hasattr(vol_spike, 'iloc') else vol_spike[i]
        
        # Regime-based logic
        if chop[i] < 38.2:  # Trending regime
            # Trade breakouts in direction of HTF trend
            if close[i] > R1[i] and htf_trend[i] == 1 and vol_spike_i:  # Long breakout
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < S1[i] and htf_trend[i] == -1 and vol_spike_i:  # Short breakout
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit on opposite touch or loss of volume spike
                if position == 1 and (close[i] < S1[i] or not vol_spike_i):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and (close[i] > R1[i] or not vol_spike_i):
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
        elif chop[i] > 61.8:  # Ranging regime
            # Mean revert at extreme touches of R1/S1
            if close[i] <= S1[i] and htf_trend[i] == 1:  # Long mean reversion at S1 in uptrend HTF
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] >= R1[i] and htf_trend[i] == -1:  # Short mean reversion at R1 in downtrend HTF
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion when price moves toward Camarilla pivot (PP)
                # PP approximated as (R1 + S1)/2
                pp = (R1[i] + S1[i]) / 2
                if position == 1 and close[i] > pp:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] < pp:
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
        else:  # Transition regime (38.2 <= CHOP <= 61.8)
            # Hold current position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0