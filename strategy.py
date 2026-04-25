#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Choppiness Filter
Hypothesis: Williams fractals on 1d identify significant swing points. A break above a bearish fractal with volume, above 1d EMA34 (uptrend), and in low choppiness (trending regime) signals bullish momentum. Conversely for short. The choppiness filter avoids whipsaws in ranging markets. 12h timeframe reduces trade frequency. Works in bull/bear via trend and regime filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter, Williams fractals, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams fractals on 1d (need 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bearish fractal needs 2 extra bars for confirmation (after center bar)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    # Bullish fractal needs 2 extra bars for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d choppiness index (CHOP) - uses 14-period
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=14).mean().values
        # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
        highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
        range_hl = highest_high - lowest_low
        chop = np.where(range_hl > 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # neutral if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34 warmup and fractal alignment
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        chop_value = chop_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Regime filter: CHOP < 50 indicates trending (lower = more trending)
        trending_regime = chop_value < 50
        
        if position == 0:
            # Long: price breaks above bearish fractal (sell fractal) AND above 1d EMA34 (uptrend filter) AND volume spike AND trending regime
            long_condition = (curr_close > bear_fractal) and (curr_close > ema_trend) and volume_spike and trending_regime
            # Short: price breaks below bullish fractal (buy fractal) AND below 1d EMA34 (downtrend filter) AND volume spike AND trending regime
            short_condition = (curr_close < bull_fractal) and (curr_close < ema_trend) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below bullish fractal or trend breaks or chop too high (rangy)
            if curr_close <= bull_fractal or curr_close < ema_trend or chop_value > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above bearish fractal or trend breaks or chop too high (rangy)
            if curr_close >= bear_fractal or curr_close > ema_trend or chop_value > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0