#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In bull trend (price > 1d EMA34): go long when Bull Power > 0 and rising, exit when Bear Power > 0.
# In bear trend (price < 1d EMA34): go short when Bear Power < 0 and falling, exit when Bull Power > 0.
# Volume > 1.5x 20-period average confirms strength. Targets 15-35 trades/year per symbol.
# Works in bull markets via trend-following longs, in bear via trend-following shorts.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close = prices['close']
    ema13 = close.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = (prices['high'] - ema13).values
    bear_power = (prices['low'] - ema13).values
    
    # 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # wait for EMA34 and EMA13
        # Skip if data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close.iloc[i]
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else 0
        br = bear_power[i]
        br_prev = bear_power[i-1] if i > 0 else 0
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend determination
        is_bull_trend = price > ema34_1d_aligned[i]
        is_bear_trend = price < ema34_1d_aligned[i]
        
        if position == 0:
            if is_bull_trend and volume_confirm:
                # Long when Bull Power positive and rising (strength increasing)
                if bp > 0 and bp > bp_prev:
                    signals[i] = 0.25
                    position = 1
            elif is_bear_trend and volume_confirm:
                # Short when Bear Power negative and falling (strength increasing)
                if br < 0 and br < br_prev:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bear Power becomes positive (bulls losing control)
                if br > 0:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit when Bull Power becomes positive (bears losing control)
                if bp > 0:
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