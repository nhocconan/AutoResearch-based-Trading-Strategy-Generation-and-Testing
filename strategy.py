#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter
# In trending regimes (ADX > 25): breakout above/below Donchian channels with volume confirmation
# In ranging regimes (ADX < 20): mean reversion at Donchian channels with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, ADX filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing
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
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1d indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below Donchian low or we enter ranging regime
                if close[i] < donchian_low[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above Donchian high
                if close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above Donchian high or we enter ranging regime
                if close[i] > donchian_high[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below Donchian low
                if close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above Donchian high with volume confirmation
                if close[i] > donchian_high[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below Donchian low with volume confirmation
                elif close[i] < donchian_low[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near Donchian low, sell near Donchian high
                if close[i] <= donchian_low[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= donchian_high[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals