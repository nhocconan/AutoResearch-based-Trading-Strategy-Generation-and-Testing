#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -20 from below + price > 12h EMA50 + volume spike
# Short when Williams %R crosses below -80 from above + price < 12h EMA50 + volume spike
# Exit when Williams %R returns to -50 level or volume drops below 80% of average.
# Williams %R captures overbought/oversold conditions; EMA50 provides trend filter.
# Works in bull (buying oversold dips) and bear (selling overbought rallies) markets.
# Target: 15-30 trades/year to avoid excessive fee drag on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for Williams %R and EMA50 calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period average on 6h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Williams %R previous value for crossover detection
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20 from below + price > EMA50 + volume spike
            if wr_prev <= -20 and wr > -20 and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -80 from above + price < EMA50 + volume spike
            elif wr_prev >= -80 and wr < -80 and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to -50 level or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to -50 or volume dries up
                if wr >= -50 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to -50 or volume dries up
                if wr <= -50 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0