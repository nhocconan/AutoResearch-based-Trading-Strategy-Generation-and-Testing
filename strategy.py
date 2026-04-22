# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA trend filter and volume spike
# Long when Williams %R crosses above -50 (bullish momentum) + close > 1d EMA34 (uptrend) + volume spike
# Short when Williams %R crosses below -50 (bearish momentum) + close < 1d EMA34 (downtrend) + volume spike
# Exit when Williams %R crosses back below -50 (long) or above -50 (short) or trend reverses
# Williams %R is a momentum oscillator measuring overbought/oversold levels.
# Designed for low trade frequency (~20-40/year) to minimize fee drain.
# Works in bull/bear by combining momentum with trend-following and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    hh_ll = highest_high - lowest_low
    willr = np.where(hh_ll != 0, ((highest_high - close) / hh_ll) * -100, -50.0)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        willr_val = willr[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R crosses above -50 + uptrend + volume spike
            if willr_val > -50.0 and willr[i-1] <= -50.0 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -50 + downtrend + volume spike
            elif willr_val < -50.0 and willr[i-1] >= -50.0 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses back or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R falls below -50 or trend turns down
                if willr_val < -50.0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R rises above -50 or trend turns up
                if willr_val > -50.0 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0