#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend filter + volume spike confirmation.
# Long when Bull Power > 0 and Bear Power < 0 in uptrend (1d EMA34 rising), short when Bear Power < 0 and Bull Power < 0 in downtrend.
# Volume > 2x 20-period average confirms momentum. Uses EMA34 to filter weak trends and avoid chop.
# Target: 12-30 trades/year by requiring strong trend + volume + Elder Ray alignment.
# Works in bull/bear: EMA34 filter ensures only strong trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend direction filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema_34_slope = np.diff(ema_34, prepend=ema_34[0])  # positive = rising
    
    # Align EMA34 slope to 6h timeframe
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # Calculate Elder Ray components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema_34_slope_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: rising EMA34 (uptrend) or falling EMA34 (downtrend)
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        if position == 0:
            if volume_confirm:
                # Long: Bull Power > 0 and Bear Power < 0 in uptrend
                if bull_power[i] > 0 and bear_power[i] < 0 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and Bull Power < 0 in downtrend
                elif bear_power[i] < 0 and bull_power[i] < 0 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Bull Power <= 0 (lost bullish momentum) or trend turns down
                if bull_power[i] <= 0 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Bear Power >= 0 (lost bearish momentum) or trend turns up
                if bear_power[i] >= 0 or not downtrend:
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