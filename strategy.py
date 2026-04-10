#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Donchian(20) from 1d: upper/lower bands for breakout detection
# - 1d EMA(50) > EMA(200) for bullish trend, < for bearish trend to align with higher timeframe
# - Volume confirmation: current 4h volume > 1.5x 20-period average to confirm breakout strength
# - Designed for 4h timeframe: targets 20-50 trades/year (75-200 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: trend filter ensures we trade with higher timeframe direction
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Includes ATR-based stoploss to manage risk

name = "4h_1d_donchian_trend_volume_v1"
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
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d EMA trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    bullish_trend = ema_50 > ema_200
    bearish_trend = ema_50 < ema_200
    
    # Align HTF indicators to LTF
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend)
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend)
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute ATR for stoploss (using 4h data)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_multiplier = 2.5  # ATR multiplier for stoploss
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or hits ATR stoploss
            if (prices['close'].iloc[i] < donchian_low_aligned[i] or 
                prices['close'].iloc[i] < entry_price - atr_multiplier * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or hits ATR stoploss
            if (prices['close'].iloc[i] > donchian_high_aligned[i] or 
                prices['close'].iloc[i] > entry_price + atr_multiplier * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above Donchian high in bullish trend
                if bullish_trend_aligned[i] and prices['close'].iloc[i] > donchian_high_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Breakout short: price closes below Donchian low in bearish trend
                elif bearish_trend_aligned[i] and prices['close'].iloc[i] < donchian_low_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals