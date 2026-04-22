#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) + 1d EMA (34) trend + volume spike.
# Williams %R identifies overbought/oversold conditions. 
# In trending markets (price > 1d EMA34), we look for pullbacks to oversold levels (Williams %R < -80) for long entries,
# and overbought levels (Williams %R > -20) for short entries, with volume confirmation.
# In ranging markets (price near 1d EMA34), we fade extremes at Williams %R < -90 (long) and > -10 (short).
# Designed to work in both bull and bear markets by adapting to trend strength.
# Targets 20-50 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA and Williams %R (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on daily close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams %R (14-period) on daily data
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align daily indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_34_aligned[i]
        wr = williams_r_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        # Trend filter: price vs daily EMA34
        is_uptrend = price > ema
        is_downtrend = price < ema
        
        if position == 0:
            if is_uptrend:
                # In uptrend: look for oversold pullback to go long
                if wr < -80 and vol_spike:
                    signals[i] = 0.30
                    position = 1
            elif is_downtrend:
                # In downtrend: look for overbought bounce to go short
                if wr > -20 and vol_spike:
                    signals[i] = -0.30
                    position = -1
            else:
                # Near EMA (ranging): fade extremes
                if wr < -90 and vol_spike:
                    signals[i] = 0.30
                    position = 1
                elif wr > -10 and vol_spike:
                    signals[i] = -0.30
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R reaches overbought or price crosses below EMA
                if wr > -20 or price < ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R reaches oversold or price crosses above EMA
                if wr < -80 or price > ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_WilliamsR_EMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0