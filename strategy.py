#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray Index with 1d EMA50 trend filter and volume spike
    # Elder Ray measures bull/bear power relative to EMA (bull power = high - EMA, bear power = low - EMA)
    # Strong bull power + bear power near zero = strong uptrend
    # Strong bear power + bull power near zero = strong downtrend
    # Combined with 1d EMA50 trend filter and volume confirmation for high-probability entries
    # Works in bull/bear by following the higher timeframe trend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull power: high minus EMA
    bear_power = low - ema13   # Bear power: low minus EMA
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x average volume
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power, bear power near zero, volume spike, price above 1d EMA50
            if bull_power[i] > 0 and bear_power[i] > -0.5 * np.std(bear_power[max(0, i-50):i+1]) and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power, bull power near zero, volume spike, price below 1d EMA50
            elif bear_power[i] < 0 and bull_power[i] < 0.5 * np.std(bull_power[max(0, i-50):i+1]) and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power divergence or trend reversal vs 1d EMA50
            if position == 1:
                if bull_power[i] < 0 or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power[i] > 0 or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0