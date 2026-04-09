#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and ADX regime filter
# In trending regimes (ADX > 25): breakout above/below Donchian(20) levels with volume confirmation
# In ranging regimes (ADX < 20): mean reversion at Donchian(20) midpoint with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, ADX filter avoids whipsaws in ranging markets

name = "4h_12h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14) for ADX
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Calculate 12h +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Calculate 12h ADX(14)
    smoothed_plus_dm = wilders_smoothing(plus_dm, 14)
    smoothed_minus_dm = wilders_smoothing(minus_dm, 14)
    smoothed_tr = wilders_smoothing(tr, 14)
    
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0.0)
    adx_12h = wilders_smoothing(dx, 14)
    
    # Calculate 12h average volume (20-period)
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian(20) channels (based on prior 20 bars to avoid look-ahead)
    highest_20 = pd.Series(high).shift(1).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).shift(1).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2
    
    # Align 12h indicators to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(midpoint_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
        volume_confirmed = volume[i] > 1.5 * avg_volume_12h_aligned[i]
        
        # Regime filter
        trending_regime = adx_12h_aligned[i] > 25
        ranging_regime = adx_12h_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below midpoint or we enter ranging regime
                if close[i] < midpoint_20[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above highest_20 or drops below lowest_20
                if close[i] > highest_20[i] or close[i] < lowest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above midpoint or we enter ranging regime
                if close[i] > midpoint_20[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below lowest_20 or rises above highest_20
                if close[i] < lowest_20[i] or close[i] > highest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above highest_20 with volume confirmation
                if close[i] > highest_20[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below lowest_20 with volume confirmation
                elif close[i] < lowest_20[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near lowest_20, sell near highest_20
                if close[i] <= lowest_20[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= highest_20[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals