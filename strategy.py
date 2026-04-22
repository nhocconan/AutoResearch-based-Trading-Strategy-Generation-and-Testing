#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Keltner Channel breakout with volume surge and weekly trend filter.
# Long when price breaks above upper KC + volume surge + weekly close > weekly EMA20
# Short when price breaks below lower KC + volume surge + weekly close < weekly EMA20
# Exit when price crosses back through weekly EMA20 or volume drops below average.
# Designed for 1d timeframe to capture multi-day trends with filtered breakouts.
# Target: 10-25 trades/year to minimize fee drag while capturing major moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Keltner Channel and EMA20
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate Keltner Channel (20, 1.5)
    # Upper = EMA20 + 1.5 * ATR(20)
    # Lower = EMA20 - 1.5 * ATR(20)
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(20) for weekly
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr20_weekly = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    upper_kc = ema20_weekly + 1.5 * atr20_weekly
    lower_kc = ema20_weekly - 1.5 * atr20_weekly
    
    # Align weekly data to daily
    upper_kc_aligned = align_htf_to_ltf(prices, df_weekly, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_weekly, lower_kc)
    ema20_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Volume surge filter (20-day average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_kc_aligned[i]) or 
            np.isnan(lower_kc_aligned[i]) or 
            np.isnan(ema20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_kc_aligned[i]
        lower = lower_kc_aligned[i]
        ema20 = ema20_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_surge = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper KC + volume surge + price > weekly EMA20
            if price > upper and vol_surge and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower KC + volume surge + price < weekly EMA20
            elif price < lower and vol_surge and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through weekly EMA20 or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below weekly EMA20 or volume drops
                if price < ema20 or vol < vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above weekly EMA20 or volume drops
                if price > ema20 or vol < vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Weekly_Keltner_Breakout_Volume_EMA20"
timeframe = "1d"
leverage = 1.0