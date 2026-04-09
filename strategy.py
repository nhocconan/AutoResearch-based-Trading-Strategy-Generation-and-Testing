#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d regime filter
# - Uses 4h ADX(14) for trend strength filter (ADX > 20 = trending)
# - Uses 1d Donchian channel breakout (20) for directional bias
# - Enters on 1h when price breaks Camarilla pivot levels (L3/H3) in direction of HTF trend
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years) to avoid fee drag
# - Combines HTF trend following with LTF pivot breakout for precision entries

name = "1h_4h_1d_camarilla_donchian_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h ADX(14) for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI and DX
    di_plus = np.divide(dm_plus_smooth, atr_4h, out=np.zeros_like(dm_plus_smooth), where=atr_4h!=0) * 100
    di_minus = np.divide(dm_minus_smooth, atr_4h, out=np.zeros_like(dm_minus_smooth), where=atr_4h!=0) * 100
    dx = np.divide(np.abs(di_plus - di_minus), (di_plus + di_minus), out=np.zeros_like(di_plus), where=(di_plus + di_minus)!=0) * 100
    adx_4h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 4h ADX to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d Donchian channel (20) for trend bias
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1h
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # 1h Camarilla pivot calculation (based on previous day)
    # We'll use rolling window of 24 1h bars to approximate 1 day
    lookback = 24  # 24 * 1h = 1 day approx
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Calculate rolling max/min/close for pivot points
    roll_high = pd.Series(high_1h).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low_1h).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close_1h).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Camarilla levels
    range_ = roll_high - roll_low
    camarilla_h3 = roll_close + (range_ * 1.1 / 4)
    camarilla_l3 = roll_close - (range_ * 1.1 / 4)
    camarilla_h4 = roll_close + (range_ * 1.1 / 2)
    camarilla_l4 = roll_close - (range_ * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(adx_4h_aligned[i]) or adx_4h_aligned[i] < 20 or  # Weak trend filter
            np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or np.isnan(roll_close[i]) or
            range_[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if close_1h[i] >= camarilla_h3[i]:  # Profit target at H3
                position = 0
                signals[i] = 0.0
            elif close_1h[i] <= roll_close[i]:  # Return to mean (daily close)
                position = 0
                signals[i] = 0.0
            elif adx_4h_aligned[i] < 20:  # Trend weakened
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if close_1h[i] <= camarilla_l3[i]:  # Profit target at L3
                position = 0
                signals[i] = 0.0
            elif close_1h[i] >= roll_close[i]:  # Return to mean (daily close)
                position = 0
                signals[i] = 0.0
            elif adx_4h_aligned[i] < 20:  # Trend weakened
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout entries aligned with HTF trend and Donchian bias
            bullish_bias = close_1h[i] > donch_low_1d_aligned[i] and close_1h[i] < donch_high_1d_aligned[i]
            bearish_bias = close_1h[i] < donch_high_1d_aligned[i] and close_1h[i] > donch_low_1d_aligned[i]
            
            # Long: price breaks above H3 with bullish HTF bias
            if (close_1h[i] > camarilla_h3[i] and 
                camarilla_h3[i] < camarilla_h4[i] and  # Valid level
                adx_4h_aligned[i] > 20 and  # Strong trend
                close_1h[i] > donch_low_1d_aligned[i]):  # Above 1d Donchian low (bullish bias)
                position = 1
                signals[i] = 0.20
            # Short: price breaks below L3 with bearish HTF bias
            elif (close_1h[i] < camarilla_l3[i] and 
                  camarilla_l3[i] > camarilla_l4[i] and  # Valid level
                  adx_4h_aligned[i] > 20 and  # Strong trend
                  close_1h[i] < donch_high_1d_aligned[i]):  # Below 1d Donchian high (bearish bias)
                position = -1
                signals[i] = -0.20
    
    return signals