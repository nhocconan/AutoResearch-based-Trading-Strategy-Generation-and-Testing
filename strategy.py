#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d ADX trend filter and 1w RSI mean reversion
# ADX > 25 indicates strong trend (use 1w RSI for mean reversion entries)
# In trending markets: buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
# In ranging markets (ADX < 20): fade moves at Bollinger Band extremes
# This adaptive approach works in both bull and bear markets by regime detection
# Uses 1d ADX for regime, 1w RSI for signals - avoids overtrading with clear regime filters

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14 periods)
    adx_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=adx_len, min_periods=adx_len).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) != 0,
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    # ADX
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 1w data ONCE for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w RSI (14 periods)
    rsi_len = 14
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Smoothed averages
    avg_gain = pd.Series(gain).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    
    # RS and RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Load 1d data for Bollinger Bands (ranging market tool)
    bb_length = 20
    bb_mult = 2.0
    bb_src = df_1d['close'].values
    
    basis = pd.Series(bb_src).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(bb_src).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = basis + dev
    lower = basis - dev
    
    # Align Bollinger Bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, adx_len * 2, rsi_len * 2, bb_length * 2)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi_aligned[i]
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        if adx_val > 25:
            # Trending market: use RSI mean reversion
            if position == 0:
                # Enter long when RSI oversold (< 30)
                if rsi_val < 30:
                    position = 1
                    signals[i] = position_size
                # Enter short when RSI overbought (> 70)
                elif rsi_val > 70:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long when RSI overbought (> 70) or neutral (> 50)
                if rsi_val > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short when RSI oversold (< 30) or neutral (< 50)
                if rsi_val < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            # Ranging market: fade at Bollinger Band extremes
            if position == 0:
                # Enter long at lower band (oversold)
                if price <= lower_band:
                    position = 1
                    signals[i] = position_size
                # Enter short at upper band (overbought)
                elif price >= upper_band:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long at upper band or middle
                if price >= upper_band or price >= (upper_band + lower_band) / 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short at lower band or middle
                if price <= lower_band or price <= (upper_band + lower_band) / 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
    
    return signals

name = "4h_1dADX_1wRSI_Adaptive_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0