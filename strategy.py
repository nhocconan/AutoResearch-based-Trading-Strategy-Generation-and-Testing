#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Williams fractals on 1w timeframe identify significant swing highs/lows.
Breakouts above bearish fractal (resistance) or below bullish fractal (support)
with 1w EMA50 trend alignment, volume confirmation, and chop regime filter
capture strong momentum moves while avoiding whipsaws in ranging markets.
12h timeframe balances trade frequency and signal quality for BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for fractal calculation and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams fractals on 1w - needs extra delay for confirmation
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Bearish fractal (swing high) needs 2 extra 1w bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    # Bullish fractal (swing low) needs 2 extra 1w bars for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d Chopiness index for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness index: CHOP = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (hh_1d - ll_1d)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, fractals, chop, and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor breakouts), chop > 61.8 = ranging (avoid)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Look for entry signals - require: breakout + trend + volume + regime
            # Long: break above bearish fractal (resistance) AND bullish bias AND volume spike AND trending
            long_entry = (curr_high > bearish_fractal_aligned[i]) and bullish_bias and vol_spike and trending_regime
            # Short: break below bullish fractal (support) AND bearish bias AND volume spike AND trending
            short_entry = (curr_low < bullish_fractal_aligned[i]) and bearish_bias and vol_spike and trending_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below EMA50 (trend change) OR re-enters fractal range
            if (curr_close < ema_1w_aligned[i]) or (curr_low > bullish_fractal_aligned[i] and curr_high < bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above EMA50 (trend change) OR re-enters fractal range
            if (curr_close > ema_1w_aligned[i]) or (curr_low > bullish_fractal_aligned[i] and curr_high < bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0