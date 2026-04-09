#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_v1
# Hypothesis: 1d strategy using Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP > 61.8 = range, mean reversion; CHOP < 38.2 = trending, trend follow). Enters long on upper band breakout with volume confirmation in trending regime; short on lower band breakout with volume confirmation in trending regime. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 7-25 trades/year) to work in both bull and bear markets by combining breakout momentum with regime adaptation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.values

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    atr = calculate_atr(high, low, close, 1)
    tr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

name = "1d_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    chop = calculate_choppiness(high, low, close, 14)
    
    # 1w HTF trend filter: 20-period EMA on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(chop[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band or regime shifts to ranging
            if close[i] < lowest_low[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band or regime shifts to ranging
            if close[i] > highest_high[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and trending regime
            if volume_confirmed and trending_regime:
                # Bullish 1w trend: price above 20-period EMA
                bullish_trend = close[i] > ema_20_1w_aligned[i]
                # Bearish 1w trend: price below 20-period EMA
                bearish_trend = close[i] < ema_20_1w_aligned[i]
                
                # Long: price breaks above upper band with volume and bullish 1w trend
                if close[i] > highest_high[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower band with volume and bearish 1w trend
                elif close[i] < lowest_low[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals