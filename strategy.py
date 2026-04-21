#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA trend filter and volume confirmation.
# In 1d uptrend (price > EMA34): look for Williams %R oversold (< -80) for long entries.
# In 1d downtrend (price < EMA34): look for Williams %R overbought (> -20) for short entries.
# Uses volume > 1.2x 20-period average for confirmation. Exit when Williams %R crosses -50.
# Target: 25-40 trades/year by requiring trend alignment + extreme readings + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
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
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirm = volume > 1.2 * vol_ma[i]
        
        # 1d trend filter
        price_above_ema = price > ema_34_1d_aligned[i]
        price_below_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # In uptrend: look for oversold conditions to go long
                if price_above_ema and williams_r[i] < -80:
                    signals[i] = 0.25
                    position = 1
                # In downtrend: look for overbought conditions to go short
                elif price_below_ema and williams_r[i] > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when Williams %R crosses -50 (mean reversion complete)
            exit_signal = False
            if position == 1 and williams_r[i] > -50:
                exit_signal = True
            elif position == -1 and williams_r[i] < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA34Trend_Volume"
timeframe = "4h"
leverage = 1.0