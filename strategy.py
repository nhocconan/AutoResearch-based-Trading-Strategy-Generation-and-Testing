#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ADX for trend strength and 1w RSI for mean reversion timing
# - Uses 1d HTF for ADX(14): ADX > 25 indicates strong trend (bullish or bearish)
# - Uses 1w HTF for RSI(14): RSI < 30 = oversold, RSI > 70 = overbought
# - In strong bullish trend (1d ADX > 25): look for long entries when 1w RSI < 30 (pullback)
# - In strong bearish trend (1d ADX > 25): look for short entries when 1w RSI > 70 (bounce)
# - No entries in weak trend (ADX <= 25) to avoid whipsaws
# - Volume confirmation: current 6h volume > 1.5x 20-period average to ensure participation
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets by only trading in direction of strong trend

name = "6h_1d_1w_adx_rsi_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d ADX (14 periods)
    # True Range
    tr1 = pd.Series(high_1d).rolling(window=2).max().values - pd.Series(low_1d).rolling(window=2).min().values
    tr2 = np.abs(pd.Series(high_1d).shift(1).values - pd.Series(close_1d).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).shift(1).values - pd.Series(close_1d).shift(1).values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Directional Movement
    dm_plus = np.where((pd.Series(high_1d).diff().values > pd.Series(low_1d).diff().values) & 
                       (pd.Series(high_1d).diff().values > 0), 
                       pd.Series(high_1d).diff().values, 0)
    dm_minus = np.where((pd.Series(low_1d).diff().values > pd.Series(high_1d).diff().values) & 
                        (pd.Series(low_1d).diff().values > 0), 
                        pd.Series(low_1d).diff().values, 0)
    
    # Smooth TR, DM+ and DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1w RSI (14 periods)
    delta = pd.Series(close_1w).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend strength: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # RSI extremes: <30 = oversold, >70 = overbought
        oversold = rsi_aligned[i] < 30
        overbought = rsi_aligned[i] > 70
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if strong_trend:
                # In strong trend: exit when RSI > 50 (mean reversion complete) or trend weakens
                if rsi_aligned[i] > 50 or not strong_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Weak trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if strong_trend:
                # In strong trend: exit when RSI < 50 (mean reversion complete) or trend weakens
                if rsi_aligned[i] < 50 or not strong_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Weak trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend strength and RSI extremes
            if volume_confirmed and strong_trend:
                if oversold:
                    # In strong trend, weekly oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif overbought:
                    # In strong trend, weekly overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals