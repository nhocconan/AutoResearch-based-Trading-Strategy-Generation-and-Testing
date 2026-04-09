#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator with 1w ADX regime filter
# Williams Alligator: Jaw (TEMA13), Teeth (TEMA8), Lips (TEMA5) - measures trend alignment
# ADX > 25 indicates trending market; ADX < 20 indicates ranging market
# In trending regime (ADX > 25): follow Alligator alignment (long when Lips>Teeth>Jaw, short when Lips<Teeth<Jaw)
# In ranging regime (ADX < 20): mean revert at extreme deviations from Alligator (long when price < Jaw - threshold, short when price > Lips + threshold)
# Uses discrete position sizing 0.25 to limit trades to ~7-25/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

name = "1d_1w_alligator_adx_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d
    # Jaw: TEMA(13), Teeth: TEMA(8), Lips: TEMA(5)
    def tema(values, period):
        # Triple Exponential Moving Average
        ema1 = pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean()
        ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
        ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
        return (3 * (ema1 - ema2) + ema3).values
    
    jaw_1d = tema(close_1d, 13)
    teeth_1d = tema(close_1d, 8)
    lips_1d = tema(close_1d, 5)
    
    # Alligator alignment signals
    # Bullish alignment: Lips > Teeth > Jaw
    # Bearish alignment: Lips < Teeth < Jaw
    bullish_alignment = (lips_1d > teeth_1d) & (teeth_1d > jaw_1d)
    bearish_alignment = (lips_1d < teeth_1d) & (teeth_1d < jaw_1d)
    
    # Price deviation from Alligator for ranging regime
    # Deviation below Jaw: price < Jaw - threshold
    # Deviation above Lips: price > Lips + threshold
    jaw_lips_avg = (jaw_1d + lips_1d) / 2
    deviation = (close_1d - jaw_lips_avg) / jaw_lips_avg  # Normalized deviation
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = wilders_smoothing(tr, 14)
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 1d timeframe (identity alignment)
    bullish_alignment_aligned = align_htf_to_ltf(prices, df_1d, bullish_alignment.astype(float))
    bearish_alignment_aligned = align_htf_to_ltf(prices, df_1d, bearish_alignment.astype(float))
    deviation_aligned = align_htf_to_ltf(prices, df_1d, deviation)
    
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Threshold for extreme deviations in ranging market
    deviation_threshold = 0.02  # 2% deviation from Jaw/Lips average
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_alignment_aligned[i]) or np.isnan(bearish_alignment_aligned[i]) or
            np.isnan(deviation_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if bullish alignment breaks
                if not bullish_alignment_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns from extreme deviation
                if deviation_aligned[i] > -deviation_threshold * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if bearish alignment breaks
                if not bearish_alignment_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns from extreme deviation
                if deviation_aligned[i] < deviation_threshold * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Follow Alligator alignment in trending market
                if bullish_alignment_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif bearish_alignment_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at extreme deviations in ranging market
                if deviation_aligned[i] < -deviation_threshold:
                    position = 1
                    signals[i] = 0.25
                elif deviation_aligned[i] > deviation_threshold:
                    position = -1
                    signals[i] = -0.25
    
    return signals