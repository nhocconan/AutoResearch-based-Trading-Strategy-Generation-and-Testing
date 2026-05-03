#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) in bull trend (close > 1d EMA34) with volume > 1.5x 20-period MA.
# Short when jaws cross below teeth in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee drag while capturing trends.
# Alligator identifies trend initiation/continuation; 1d EMA34 filter ensures alignment with higher timeframe trend.
# Volume confirmation filters weak breakouts. Target: 80-160 total trades over 4 years (20-40/year).

name = "4h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as Wilder's EMA with alpha=1/length"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) == 0:
        return result
    result[0] = source[0]
    for i in range(1, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h: Jaw (13), Teeth (8), Lips (5) - all SMMA of median price
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)   # Red line
    lips = smma(median_price, 5)    # Green line (not used in signals)
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime from 1d EMA34
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Alligator crossover signals
        # Bullish: Jaw crosses above Teeth
        bullish_cross = (jaw_val > teeth_val) and (jaw[i-1] <= teeth[i-1]) if i > 0 else False
        # Bearish: Jaw crosses below Teeth
        bearish_cross = (jaw_val < teeth_val) and (jaw[i-1] >= teeth[i-1]) if i > 0 else False
        
        # Entry logic
        if position == 0:
            if is_bull_trend and bullish_cross and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and bearish_cross and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish crossover OR trend reversal
            if bearish_cross or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish crossover OR trend reversal
            if bullish_cross or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals