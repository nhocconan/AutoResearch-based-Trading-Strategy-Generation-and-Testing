#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator with 1d regime filter
# - Bull Power = High - EMA13, Bear Power = EMA8 - Low
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Regime: ADX(14) > 25 for trending, < 20 for ranging (from 1d)
# - Long: Bull Power > 0 AND Bear Power < 0 AND Lips > Teeth > Jaw (bullish alignment) AND trending regime
# - Short: Bull Power < 0 AND Bear Power > 0 AND Lips < Teeth < Jaw (bearish alignment) AND trending regime
# - Exit: Power signals weaken (Bull Power < 0 for long, Bear Power > 0 for short) OR regime shifts to ranging
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: captures strong trends via Elder Ray power and Alligator alignment, avoids chop with regime filter

name = "6h_1d_elder_ray_alligator_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate EMA8 and EMA13 for Elder Ray
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA8 - Low
    bull_power = high - ema_13
    bear_power = ema_8 - low
    
    # Williams Alligator components on median price
    median_price = (high + low) / 2.0
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using EMA as approximation for SMMA (common practice)
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    plus_di = np.where(tr_14 == 0, 0, plus_di)
    minus_di = np.where(tr_14 == 0, 0, minus_di)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_8[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        # Williams Alligator alignment
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        bullish_align = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_align = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Entry conditions
        long_entry = bull_strong and (not bear_strong) and bullish_align and is_trending
        short_entry = bear_strong and (not bull_strong) and bearish_align and is_trending
        
        # Exit conditions: power weakens OR regime shifts to ranging
        exit_long = (bull_power[i] <= 0) or is_ranging
        exit_short = (bear_power[i] <= 0) or is_ranging
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals