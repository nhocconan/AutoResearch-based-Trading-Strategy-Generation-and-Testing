#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d regime filter
# Uses Williams Alligator (jaw/teeth/lips) to identify trending vs ranging markets
# Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13
# 1d ADX > 25 confirms trending regime for Alligator signals
# Only trade when Alligator is aligned (trending) and Elder Ray confirms direction
# Designed for ~15-30 trades/year to minimize fee drag while capturing strong trends
# Works in bull/bear via regime filter - avoids false signals in ranging markets

name = "6h_WilliamsAlligator_ElderRay_1dADX25_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter (trending >25)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    def smma(source, period):
        # Smoothed Moving Average (similar to Wilder's smoothing)
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(high, 13)  # Using high for jaw (alligator typically uses median price)
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    # Shift the lines (Alligator-specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Calculate Elder Ray on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        is_trending = curr_adx > 25
        
        # Alligator alignment: 
        # Bullish alignment: Lips > Teeth > Jaw (all rising)
        # Bearish alignment: Jaw > Teeth > Lips (all falling)
        bullish_aligned = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_aligned = curr_jaw > curr_teeth and curr_teeth > curr_lips
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Elder Ray turns negative or Alligator loses alignment
            if curr_bull_power <= 0 or not bullish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray turns positive or Alligator loses alignment
            if curr_bear_power >= 0 or not bearish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter in trending markets
            if is_trending:
                # Long when Alligator bullish aligned AND Bull Power positive AND rising
                if bullish_aligned and curr_bull_power > 0 and curr_bull_power > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short when Alligator bearish aligned AND Bear Power negative AND falling (more negative)
                elif bearish_aligned and curr_bear_power < 0 and curr_bear_power < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Stay flat in ranging markets
    
    return signals