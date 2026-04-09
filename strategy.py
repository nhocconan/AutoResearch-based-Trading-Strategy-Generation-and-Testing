#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h Donchian breakout with volume confirmation (>1.5x 20-bar avg volume) and chop regime filter (CHOP<61.8 = trending). Uses 12h HTF EMA(50) for trend alignment. Discrete position sizing (0.25) to minimize fee churn. Tightened entry conditions by requiring close > previous Donchian high/low (not just touching) and added minimum holding period of 3 bars to reduce churn. Target: 19-50 trades/year (75-200 total over 4 years). Works in bull/bear: Donchian captures breakouts, volume confirms conviction, chop filter avoids whipsaws in ranging markets, HTF EMA ensures alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    # Avoid division by zero or log of zero
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema_50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track holding period
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # HTF trend filter: price above/below 12h EMA(50)
        htf_uptrend = close[i] > ema_50_12h_aligned[i]
        htf_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit: price closes below Donchian low (20) OR minimum holding period met with reversal signal
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif bars_since_entry >= 3 and close[i] < donchian_high[i-1]:  # Early exit on pullback
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit: price closes above Donchian high (20) OR minimum holding period met with reversal signal
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif bars_since_entry >= 3 and close[i] > donchian_low[i-1]:  # Early exit on pullback
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
        else:  # Flat
            bars_since_entry = 0
            # Check for Donchian breakout with volume, regime, and HTF confirmation
            # Require close to exceed previous Donchian level (not just touch)
            bullish_breakout = (close[i] > donchian_high[i-1]) and volume_confirmed and trending_market and htf_uptrend
            bearish_breakout = (close[i] < donchian_low[i-1]) and volume_confirmed and trending_market and htf_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals