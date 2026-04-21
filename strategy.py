#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-day trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (price above/below 1-day EMA34),
# we take counter-trend entries at extremes: short when Williams %R > -20 (overbought) in uptrend,
# long when Williams %R < -80 (oversold) in downtrend. Volume > 1.5x 20-period average confirms momentum.
# This strategy aims for 12-30 trades/year by requiring Williams %R extremes + trend alignment + volume confirmation.
# Works in both bull and bear markets by using 1-day trend to determine direction.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1-day HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 12h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    close = prices['close'].values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend from 1-day EMA34
        price_above_ema = price > ema_34_1d_aligned[i]
        price_below_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # In uptrend (price above 1-day EMA34): look for overbought to short
                if price_above_ema and williams_r[i] > -20:
                    signals[i] = -0.25
                    position = -1
                # In downtrend (price below 1-day EMA34): look for oversold to long
                elif price_below_ema and williams_r[i] < -80:
                    signals[i] = 0.25
                    position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long when Williams %R returns to neutral (> -50) or stops oversold
                if williams_r[i] > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit short when Williams %R returns to neutral (< -50) or stops overbought
                if williams_r[i] < -50:
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