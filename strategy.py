#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ADX for trend strength and 1w RSI for mean reversion
# - Uses 1d HTF for ADX: ADX > 25 indicates strong trend (bullish/bearish)
# - Uses 1w HTF for RSI: extreme readings (>70 or <30) signal mean reversion opportunities
# - In strong bullish trend (ADX>25 + DI+ > DI-): look for long entries when weekly RSI < 30 (oversold pullback)
# - In strong bearish trend (ADX>25 + DI- > DI+): look for short entries when weekly RSI > 70 (overbought bounce)
# - Volume confirmation: current 6h volume > 1.2x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

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
    tr2 = np.abs(pd.Series(high_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff().values
    down_move = -pd.Series(low_1d).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w RSI (14 periods)
    delta = pd.Series(close_1w).diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Trend strength and direction
        strong_trend = adx_aligned[i] > 25
        bullish_trend = strong_trend and (plus_di_aligned[i] > minus_di_aligned[i])
        bearish_trend = strong_trend and (minus_di_aligned[i] > plus_di_aligned[i])
        
        # RSI extremes: <30 = oversold, >70 = overbought
        oversold = rsi_aligned[i] < 30
        overbought = rsi_aligned[i] > 70
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when overbought or trend weakens/reverses
                if overbought or not bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when oversold or trend weakens/reverses
                if oversold or not bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and RSI extremes
            if volume_confirmed:
                if bullish_trend and oversold:
                    # In strong bullish trend, weekly oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and overbought:
                    # In strong bearish trend, weekly overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals