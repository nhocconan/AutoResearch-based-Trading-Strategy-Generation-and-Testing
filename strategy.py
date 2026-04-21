#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA34 trend + volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (price > EMA34),
# we look for pullbacks to oversold levels to go long, and in downtrends (price < EMA34),
# we look for pullbacks to overbought levels to go short. Volume confirms the pullback strength.
# This strategy aims to capture mean-reversion within the trend, reducing whipsaws.
# Target: 15-35 trades/year by requiring trend alignment, Williams %R extremes, and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d (14-period)
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d_arr = close_1d.values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d_arr) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend direction from 1d EMA34
        price_above_ema = price > ema_34_1d_aligned[i]
        price_below_ema = price < ema_34_1d_aligned[i]
        
        # Williams %R levels
        wr = williams_r_aligned[i]
        oversold = wr <= -80
        overbought = wr >= -20
        
        if position == 0:
            if volume_confirm:
                # In uptrend, look for oversold pullback to go long
                if price_above_ema and oversold:
                    signals[i] = 0.25
                    position = 1
                # In downtrend, look for overbought pullback to go short
                elif price_below_ema and overbought:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses above EMA34 (trend strength) or Williams %R becomes overbought
                if price >= ema_34_1d_aligned[i] or wr >= -20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses below EMA34 (trend strength) or Williams %R becomes oversold
                if price <= ema_34_1d_aligned[i] or wr <= -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0