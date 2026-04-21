#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, and price > 1d EMA34 in uptrend.
# Short when Bear Power < 0, Bull Power < 0, and price < 1d EMA34 in downtrend.
# Volume > 1.5x 20-period average confirms strength. Target: 20-40 trades/year.
# Works in bull/bear: Elder Ray captures bull/bear power, EMA34 filter avoids counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Elder Ray (Bull Power, Bear Power) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bull Power = High - EMA(13)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    
    # Bear Power = Low - EMA(13)
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # start after EMA34 warmup
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > 1d EMA34, volume confirmation
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                price > ema_34_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power < 0, price < 1d EMA34, volume confirmation
            elif (bear_power_aligned[i] < 0 and bull_power_aligned[i] < 0 and 
                  price < ema_34_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Bull Power <= 0 (bullish momentum fading) or price < 1d EMA34
                if bull_power_aligned[i] <= 0 or price < ema_34_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Bear Power >= 0 (bearish momentum fading) or price > 1d EMA34
                if bear_power_aligned[i] >= 0 or price > ema_34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34Trend_Volume"
timeframe = "6h"
leverage = 1.0