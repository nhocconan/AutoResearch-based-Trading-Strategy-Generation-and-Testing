#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA34 trend filter and volume confirmation.
# Bull Power (BP) = High - EMA13; Bear Power (BP) = Low - EMA13.
# In 1d uptrend (close > EMA34): long when BP > 0 and volume > 1.5x 20-period average.
# In 1d downtrend (close < EMA34): short when BP < 0 and volume > 1.5x 20-period average.
# Elder Ray captures buying/selling pressure relative to trend; 1d EMA34 filters for primary trend.
# Volume confirmation avoids low-conviction moves. Target: 12-37 trades/year on 6h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(prices['close'])
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = prices['high'].values - ema13
    bear_power = prices['low'].values - ema13
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        bp = bull_power[i]
        bb = bear_power[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # 1d trend
        is_uptrend = price > ema34_1d_aligned[i]
        is_downtrend = price < ema34_1d_aligned[i]
        
        if position == 0:
            if is_uptrend and volume_confirm:
                # Long when bull power positive (buying pressure)
                if bp > 0:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and volume_confirm:
                # Short when bear power negative (selling pressure)
                if bb < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power turns negative or trend changes
                if bp <= 0 or not is_uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bear power turns positive or trend changes
                if bb >= 0 or not is_downtrend:
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