#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA trend filter + volume confirmation
# Williams Alligator uses smoothed median prices (Jaw=13, Teeth=8, Lips=5)
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA34 + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA34 + volume spike
# Exit when Alligator alignment breaks or price crosses 1d EMA34
# Designed for low frequency (15-30 trades/year) with trend-following edge in all markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data (median price = (high+low)/2)
    median_price = (prices['high'] + prices['low']) / 2
    median_price = median_price.values
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Calculate volume spike using 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_34_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: bullish alignment + uptrend + volume spike
            if bullish_alignment and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + downtrend + volume spike
            elif bearish_alignment and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or price crosses EMA
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or price crosses below EMA
                if not bullish_alignment or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or price crosses above EMA
                if not bearish_alignment or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0