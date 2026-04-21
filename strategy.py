#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d EMA34 trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum), short when Bear Power > 0 and Bull Power < 0 (bearish momentum).
# Trend filter: 1d EMA34 slope > 0 for long, < 0 for short.
# Volume confirmation: volume > 1.5x 20-period average.
# Works in bull/bear: EMA34 filter ensures trading with higher timeframe trend, avoiding counter-trend whipsaws.
# Target: 15-30 trades/year by requiring Elder Ray alignment + trend + volume.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema_34_slope = np.diff(ema_34, prepend=ema_34[0])
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = prices['high'].values - ema_13
    bear_power = prices['low'].values - ema_13
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema_34_slope_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        bull = bull_power[i]
        bear = bear_power[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Bull Power > 0 and Bear Power < 0 (bullish) + uptrend (EMA34 slope > 0)
                if bull > 0 and bear < 0 and ema_34_slope_aligned[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 and Bull Power < 0 (bearish) + downtrend (EMA34 slope < 0)
                elif bear > 0 and bull < 0 and ema_34_slope_aligned[i] < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Bear Power turns positive (loss of bullish momentum) or trend turns down
                if bear >= 0 or ema_34_slope_aligned[i] < 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Bull Power turns positive (loss of bearish momentum) or trend turns up
                if bull >= 0 or ema_34_slope_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0