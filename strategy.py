#!/usr/bin/env python3
# 1d_keltner_breakout_volume_regime_v1
# Hypothesis: 1d strategy using Keltner Channel breakouts with volume confirmation and chop regime filter.
# Long when price breaks above upper Keltner (EMA20 + 2*ATR10) with volume > 1.5x 20-day average and chop > 61.8 (ranging).
# Short when price breaks below lower Keltner (EMA20 - 2*ATR10) with volume > 1.5x 20-day average and chop > 61.8 (ranging).
# Exit when price crosses back through EMA20.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: Keltner breakouts capture volatility expansion, volume confirms conviction, chop filter avoids whipsaws in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_breakout_volume_regime_v1"
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
    
    # EMA20 for Keltner middle line
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR10 for Keltner width
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr10 = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Keltner Channels
    upper_keltner = ema20 + 2 * atr10
    lower_keltner = ema20 - 2 * atr10
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period) - chop > 61.8 = ranging (good for mean reversion at extremes)
    atr_period = 14
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion from extremes)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA20
            if close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA20
            if close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Keltner breakout with volume and regime confirmation
            bullish_breakout = (close[i] > upper_keltner[i] and close[i-1] <= upper_keltner[i-1]) and volume_confirmed and ranging_market
            bearish_breakout = (close[i] < lower_keltner[i] and close[i-1] >= lower_keltner[i-1]) and volume_confirmed and ranging_market
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals