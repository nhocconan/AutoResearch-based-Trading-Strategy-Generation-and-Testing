#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA34 trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (price > EMA34),
# we take pullbacks: long when %R crosses above -80 from below, short when %R crosses below -20 from above.
# In ranging markets, we fade extremes: long when %R crosses above -80, short when %R crosses below -20.
# Volume confirmation ensures institutional participation. Designed for 12h timeframe to target 12-37 trades/year.
# Works in both bull (trend following) and bear (mean reversion) markets via regime detection.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA34 (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R on 12h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
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
        ema = ema_34_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        # Determine market regime based on price vs EMA34
        is_uptrend = price > ema
        is_downtrend = price < ema
        
        if position == 0:
            # Williams %R crossover signals
            wr_cross_above_80 = (wr > -80) and (i == 0 or williams_r[i-1] <= -80)
            wr_cross_below_20 = (wr < -20) and (i == 0 or williams_r[i-1] >= -20)
            
            if is_uptrend:
                # Uptrend: look for pullbacks to go long
                if wr_cross_above_80 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # In strong uptrend, also consider shorting at overbought
                elif wr_cross_below_20 and vol_spike:
                    signals[i] = -0.20
                    position = -1
            elif is_downtrend:
                # Downtrend: look for bounces to go short
                if wr_cross_below_20 and vol_spike:
                    signals[i] = -0.25
                    position = -1
                # In strong downtrend, also consider longing at oversold
                elif wr_cross_above_80 and vol_spike:
                    signals[i] = 0.20
                    position = 1
            else:
                # Ranging: fade extremes
                if wr_cross_above_80 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif wr_cross_below_20 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on Williams %R crossing below -50 (momentum loss) or opposite extreme
                if wr < -50 or wr_cross_below_20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on Williams %R crossing above -50 (momentum loss) or opposite extreme
                if wr > -50 or wr_cross_above_80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_EMA34_Volume"
timeframe = "12h"
leverage = 1.0