#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Bollinger Band width squeeze with 1-day trend filter and volume confirmation.
# Enters long when price closes above upper Bollinger Band with daily uptrend and volume spike,
# short when price closes below lower Bollinger Band with daily downtrend and volume spike.
# Exits on trend reversal or price crossing opposite Bollinger Band.
# Bollinger Band width acts as volatility filter - low width indicates consolidation before breakout.
# Designed to work in both bull and bear markets by aligning with daily trend.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_BollingerWidth_Squeeze_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 1d close (20-period, 2 std dev)
    close_1d = df_1d['close'].values
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_1d + (2 * std20_1d)
    lower_bb = sma20_1d - (2 * std20_1d)
    
    # Calculate EMA20 on 1d close for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for BB and EMA20 (1d)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        ema20_1d_val = ema20_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close above upper BB + 1d uptrend + volume spike
            if close[i] > upper_bb_val and close[i] > ema20_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close below lower BB + 1d downtrend + volume spike
            elif close[i] < lower_bb_val and close[i] < ema20_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close below lower BB or 1d trend turns down
            if close[i] < lower_bb_val or close[i] < ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close above upper BB or 1d trend turns up
            if close[i] > upper_bb_val or close[i] > ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals