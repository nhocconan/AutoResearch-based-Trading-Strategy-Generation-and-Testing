#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter
# In trending regimes (ADX > 25): breakout above/below 20-period Donchian channels with volume confirmation
# In ranging regimes (ADX < 20): mean reversion at Donchian channel midlines with volume confirmation
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and reduce fee drag
# Works in bull/bear markets: breakout captures trends, ADX filter avoids whipsaws in ranging markets

name = "12h_1d_donchian_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for ADX
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
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
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d +DM and -DM for ADX
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed +DM, -DM, and TR
    atr_1d_smooth = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Calculate 1d ADX(14)
    plus_di = 100 * plus_dm_smooth / atr_1d_smooth
    minus_di = 100 * minus_dm_smooth / atr_1d_smooth
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Calculate 1d Donchian channels (20-period) based on prior day to avoid look-ahead
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midline_20 = (highest_20 + lowest_20) / 2
    
    # Align 1d indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    midline_20_aligned = align_htf_to_ltf(prices, df_1d, midline_20)
    
    # Pre-compute volume confirmation array
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(highest_20_aligned[i]) or
            np.isnan(lowest_20_aligned[i]) or np.isnan(midline_20_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price falls below midline or we enter ranging regime
                if close[i] < midline_20_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above upper channel or drops below lower channel
                if close[i] > highest_20_aligned[i] or close[i] < lowest_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price rises above midline or we enter ranging regime
                if close[i] > midline_20_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below lower channel or rises above upper channel
                if close[i] < lowest_20_aligned[i] or close[i] > highest_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above upper channel with volume confirmation
                if close[i] > highest_20_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below lower channel with volume confirmation
                elif close[i] < lowest_20_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near lower channel, sell near upper channel
                if close[i] <= lowest_20_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= highest_20_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals