#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R momentum with 1d EMA34 trend filter and volume spike.
# Williams %R measures momentum on a scale of -100 to 0. Readings below -80 indicate oversold,
# above -20 overbought. We buy when %R crosses above -80 from below in an uptrend (price > EMA34)
# with volume confirmation (>2x 20-period avg). Sell when %R crosses below -20 from above in a
# downtrend (price < EMA34) with volume confirmation. This captures mean-reversion bounces
# within the prevailing trend, optimized for low trade frequency to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Williams %R crossovers with memory of previous value
        if i > 0:
            wr_prev = williams_r[i-1]
        else:
            wr_prev = wr
        
        wr_cross_above_80 = (wr_prev <= -80) and (wr > -80)
        wr_cross_below_20 = (wr_prev >= -20) and (wr < -20)
        
        if position == 0:
            # Long: WR crosses above -80 from below + uptrend + volume spike
            if wr_cross_above_80 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: WR crosses below -20 from above + downtrend + volume spike
            elif wr_cross_below_20 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when WR crosses below -20 (overbought) or trend breaks
                if wr_cross_below_20 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when WR crosses above -80 (oversold) or trend breaks
                if wr_cross_above_80 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_14_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0