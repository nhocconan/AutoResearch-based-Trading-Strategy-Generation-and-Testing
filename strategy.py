#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) in uptrend (1d EMA34 rising).
# Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) in downtrend (1d EMA34 falling).
# Volume > 2x 20-period average confirms momentum strength.
# Works in bull/bear: EMA34 trend filter ensures trading with dominant trend, avoiding counter-trend whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev  # uptrend
    ema_34_falling = ema_34 < ema_34_prev  # downtrend
    
    # Align EMA34 trend to 6h timeframe
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Calculate EMA13 for Elder Ray on 6h data
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or \
           np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Bull Power > 0 and Bear Power < 0 (bullish momentum) in uptrend
                if bull_power[i] > 0 and bear_power[i] < 0 and ema_34_rising_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 and Bull Power < 0 (bearish momentum) in downtrend
                elif bear_power[i] > 0 and bull_power[i] < 0 and ema_34_falling_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if bullish momentum fails (Bear Power >= 0) or trend changes
                if bear_power[i] >= 0 or not ema_34_rising_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if bearish momentum fails (Bull Power >= 0) or trend changes
                if bull_power[i] >= 0 or not ema_34_falling_aligned[i]:
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