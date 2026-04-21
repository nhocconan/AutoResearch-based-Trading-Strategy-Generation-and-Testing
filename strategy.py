#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong 1d trends (price > EMA50),
# we fade extremes only when aligned with trend (pullbacks in uptrend, bounces in downtrend).
# Volume > 1.5x 20-period average confirms conviction. Target: 20-40 trades/year.
# Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Oversold: < -80, Overbought: > -20

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1. Williams %R (14-period)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # 2. 1d EMA50 trend filter (using mtf_data)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 3. Volume confirmation (20-period average)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        wr = williams_r[i]
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend determination from 1d EMA50
        uptrend = price > ema_50
        downtrend = price < ema_50
        
        if position == 0:
            if volume_confirm:
                # In uptrend: look for oversold pullbacks to go long
                if uptrend and wr < -80:
                    signals[i] = 0.25
                    position = 1
                # In downtrend: look for overbought bounces to go short
                elif downtrend and wr > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to overbought (profit target) 
                # or breaks below -80 (failed bounce)
                if wr > -20 or wr < -85:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to oversold (profit target)
                # or breaks above -20 (failed breakdown)
                if wr < -80 or wr > -15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0