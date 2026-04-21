#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold) in uptrend (price > 1d EMA34), 
# Short when crosses below -20 (overbought) in downtrend (price < 1d EMA34).
# Volume > 1.3x 20-period average confirms momentum. Target: 20-40 trades/year.
# Works in bull/bear: EMA filter ensures trades align with higher timeframe trend,
# avoiding counter-trend entries in chop.

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
    
    # Calculate Williams %R (14-period) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = np.where((highest_high - lowest_low) != 0, 
                  ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                  -50)  # neutral when range is zero
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(wr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA34
        uptrend = price > ema_34_aligned[i]
        downtrend = price < ema_34_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -80 (oversold) in uptrend
                if wr[i] > -80 and wr[i-1] <= -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought) in downtrend
                elif wr[i] < -20 and wr[i-1] >= -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R crosses below -50 (momentum loss) or trend reversal
                if wr[i] < -50 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R crosses above -50 (momentum loss) or trend reversal
                if wr[i] > -50 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0