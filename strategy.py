# Hypothesis: 4h Camarilla Pivot Point S1/R1 breakout with 12h trend filter and volume spike
# Camarilla levels act as strong support/resistance; breakouts with volume and trend confirmation yield high-probability trades.
# Works in bull (breakouts up) and bear (breakouts down) markets due to symmetry.
# Target: ~30-60 trades/year to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla pivot levels for previous day (using prior day's OHLC)
    # Shift by 1 to avoid look-ahead: use previous day's data for today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (shifted by 1)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    # First value will be invalid (roll wraps), but we'll handle with min_periods later
    
    # Camarilla formulas
    range_ = phigh - plow
    camarilla_r1 = pclose + range_ * 1.1 / 12
    camarilla_r2 = pclose + range_ * 1.1 / 6
    camarilla_r3 = pclose + range_ * 1.1 / 4
    camarilla_r4 = pclose + range_ * 1.1 / 2
    camarilla_s1 = pclose - range_ * 1.1 / 12
    camarilla_s2 = pclose - range_ * 1.1 / 6
    camarilla_s3 = pclose - range_ * 1.1 / 4
    camarilla_s4 = pclose - range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Price and volume arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (to reduce trades)
        vol_spike = vol > 2.0 * vol_ma
        
        # Trend filter: price above/below 12h EMA50
        uptrend = price > ema50_12h_val
        downtrend = price < ema50_12h_val
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + uptrend + volume spike
            if price > r1 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + downtrend + volume spike
            elif price < s1 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Camarilla level or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on break below Camarilla S1 or volume drop
                if price < s1 or not vol_spike:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on break above Camarilla R1 or volume drop
                if price > r1 or not vol_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0