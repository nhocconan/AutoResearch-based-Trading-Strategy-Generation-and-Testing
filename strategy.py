#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams fractals on 1d identify significant swing points. A break above a bearish fractal with volume and 1d uptrend (EMA34) signals bullish momentum; break below a bullish fractal with volume and 1d downtrend signals bearish momentum. Added choppiness index (CHOP) regime filter: only trade when CHOP < 61.8 (trending market) to avoid whipsaws in ranging conditions. 12h timeframe minimizes fee drag while capturing multi-day swings. Works in both bull/bear markets via trend filter and regime adaptation.
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
    
    # Get 1d data for EMA34 trend filter and Williams fractals
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
    
    # Calculate 12h choppiness index for regime filter
    chop_period = 14
    if n >= chop_period:
        # True range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Sum of true range over chop_period
        tr_sum = np.zeros(n)
        for i in range(chop_period, n):
            tr_sum[i] = np.nansum(tr[i-chop_period+1:i+1])
        
        # Highest high and lowest low over chop_period
        hh = np.zeros(n)
        ll = np.zeros(n)
        for i in range(chop_period-1, n):
            hh[i] = np.max(high[i-chop_period+1:i+1])
            ll[i] = np.min(low[i-chop_period+1:i+1])
        
        # Chop = 100 * log10(sum(tr)/(hh-ll)) / log10(chop_period)
        chop = np.full(n, np.nan)
        for i in range(chop_period, n):
            if hh[i] > ll[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(chop_period)
            else:
                chop[i] = 50.0  # avoid division by zero
    else:
        chop = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34 warmup and fractal alignment
    start_idx = max(34, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        curr_chop = chop[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        if curr_chop >= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above bearish fractal (sell fractal) AND above 1d EMA34 (uptrend filter)
            long_condition = (curr_close > bear_fractal) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below bullish fractal (buy fractal) AND below 1d EMA34 (downtrend filter)
            short_condition = (curr_close < bull_fractal) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below bullish fractal or trend breaks
            if curr_close <= bull_fractal or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above bearish fractal or trend breaks
            if curr_close >= bear_fractal or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0