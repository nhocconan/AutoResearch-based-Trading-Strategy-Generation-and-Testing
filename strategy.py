#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold) in uptrend (price > 1d EMA34), short when crosses below -20 (overbought) in downtrend.
# Volume > 1.3x 20-period average confirms momentum. EMA filter avoids counter-trend trades.
# Target: 20-40 trades/year by requiring oversold/overbought extremes + trend alignment.
# Works in bull/bear: EMA filter ensures trades with trend, Williams %R captures mean reversion within trend.

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
    
    # Calculate Williams %R(14) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
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
                # Long: Williams %R crosses above -80 from oversold in uptrend
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from overbought in downtrend
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R crosses below -50 (momentum fading) or trend reversal
                if williams_r[i] < -50 or (price < ema_34_aligned[i] and williams_r[i] < -30):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R crosses above -50 (momentum fading) or trend reversal
                if williams_r[i] > -50 or (price > ema_34_aligned[i] and williams_r[i] > -70):
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