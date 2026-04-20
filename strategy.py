#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
# - Trend filter: 1w EMA50 - price above EMA50 = bullish, below = bearish
# - Volume: current volume > 1.5x 20-period average for confirmation
# - Long: Lips > Jaw > Teeth (bullish alignment) + price > Lips + volume confirmation
# - Short: Lips < Jaw < Teeth (bearish alignment) + price < Lips + volume confirmation
# - Exit: when Alligator alignment breaks or volume drops below average
# - Uses 1w for trend filter (major trend) and 12h for execution
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load 12h data for Williams Alligator and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(sma) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(series)):
                if not np.isnan(smma_vals[i-1]) and not np.isnan(sma[i]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
                else:
                    smma_vals[i] = np.nan
        return smma_vals
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(lips[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: Bullish Alligator alignment + price above Lips + volume confirmation + price above 1w EMA50
            if (lips[i] > jaw[i] and jaw[i] > teeth[i] and  # Bullish alignment: Lips > Jaw > Teeth
                price > lips[i] and 
                vol > 1.5 * vol_ma[i] and
                price > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment + price below Lips + volume confirmation + price below 1w EMA50
            elif (lips[i] < jaw[i] and jaw[i] < teeth[i] and  # Bearish alignment: Lips < Jaw < Teeth
                  price < lips[i] and
                  vol > 1.5 * vol_ma[i] and
                  price < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR volume drops below average OR price crosses below Lips
            if not (lips[i] > jaw[i] and jaw[i] > teeth[i]) or vol < vol_ma[i] or price < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR volume drops below average OR price crosses above Lips
            if not (lips[i] < jaw[i] and jaw[i] < teeth[i]) or vol < vol_ma[i] or price > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0