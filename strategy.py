#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
# - Donchian(20) breakout provides objective entry structure
# - 1d EMA200 trend filter ensures alignment with long-term trend (works in bull/bear regimes)
# - Volume confirmation (current volume > 2.0x 20-period average) filters weak breakouts
# - Exit: Donchian(10) opposite touch or ATR-based stoploss (2.0*ATR)
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee churn
# - Target trade frequency: 20-40 trades/year on 4h (80-160 total over 4 years)
# - Works in both bull and bear: EMA200 adapts to regime, volume reduces false signals, Donchian provides clear breakout levels

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian(20) for breakout
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d close for trend filter (aligned)
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # 1d trend filter: price > EMA200 = bullish, price < EMA200 = bearish
        bullish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] < trend_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Donchian(20) high AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > donchian_high_20[i] and bullish_trend and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Donchian(20) low AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < donchian_low_20[i] and bearish_trend and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Exit conditions: Donchian(10) opposite touch
            exit_long = prices['close'].iloc[i] < donchian_low_10[i]   # Price breaks below Donchian(10) low (exit long)
            exit_short = prices['close'].iloc[i] > donchian_high_10[i]  # Price breaks above Donchian(10) high (exit short)
            
            # Stoploss conditions: ATR-based (2.0 * ATR)
            if position == 1:
                stoploss_hit = prices['close'].iloc[i] < entry_price - 2.0 * atr[i]
            else:  # position == -1
                stoploss_hit = prices['close'].iloc[i] > entry_price + 2.0 * atr[i]
            
            exit_condition = exit_long or exit_short or stoploss_hit
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals