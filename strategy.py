#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Fractals on 1d identify significant swing highs/lows. 
Break above bullish fractal with volume and 1d EMA34 uptrend signals bullish momentum.
Break below bearish fractal with volume and 1d EMA34 downtrend signals bearish momentum.
Choppiness index filter avoids whipsaws in ranging markets. Uses 12h timeframe for lower trade frequency.
Volume spike confirms institutional participation. Target: 12-37 trades/year.
Works in bull/bear via EMA34 trend filter and chop regime adaptation.
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
    
    # Get 1d data for Williams fractals, EMA34 trend, and choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate 1d Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and low[n-2] < low[n-1] > low[n]
    # Bullish fractal: high[n-2] > high[n-1] < high[n] and low[n-2] > low[n-1] < low[n]
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Williams fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR1, n) / (max(high, n) - min(low, n))) / log10(n)
    # where n=14 period
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=14).sum().values  # sum for CHOP formula
        
        # Rolling max(high) and min(low) over 14 periods
        max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        range_14 = max_high_14 - min_low_14
        
        # Avoid division by zero
        chop_14 = np.where(
            range_14 != 0,
            100 * np.log10(atr_14 / range_14) / np.log10(14),
            50  # neutral value when range is zero
        )
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    else:
        chop_aligned = np.full(n, 50.0)  # default to neutral chop if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        bull_fractal = bullish_fractal_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        ema_34 = ema_34_aligned[i]
        chop_value = chop_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter: price above/below EMA34
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        # Chop filter: CHOP > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
        # We'll use CHOP < 50 as our filter to avoid strong ranging markets
        not_choppy = chop_value < 50
        
        if position == 0:
            # Long: price breaks above bullish fractal AND volume spike AND uptrend AND not choppy
            long_condition = (curr_high > bull_fractal) and volume_spike and uptrend and not_choppy
            # Short: price breaks below bearish fractal AND volume spike AND downtrend AND not choppy
            short_condition = (curr_low < bear_fractal) and volume_spike and downtrend and not_choppy
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below bearish fractal or trend changes or choppy market
            if curr_close <= bear_fractal or not uptrend or chop_value > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above bullish fractal or trend changes or choppy market
            if curr_close >= bull_fractal or not downtrend or chop_value > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0