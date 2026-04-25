#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 12h EMA50 Trend + Volume Spike + Chop Regime Filter
Hypothesis: Williams Fractals identify swing points where price reverses. Breakouts above bearish fractals (sell signals flipped) or below bullish fractals (buy signals flipped) with 12h EMA50 trend alignment, volume confirmation, and non-choppy regime capture momentum after reversal confirmation. Works in bull/bear markets via discrete sizing (0.25) and trend filter. Uses 12h timeframe for HTF filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (sell signal) and bullish (buy signal)"""
    n = len(high)
    bearish = np.full(n, np.nan)  # sell signal fractal
    bullish = np.full(n, np.nan)  # buy signal fractal
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest among 5 bars (i-2, i-1, i, i+1, i+2)
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest among 5 bars (i-2, i-1, i, i+1, i+2)
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50, Chop regime, and Williams Fractals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h Chop regime filter: CHOP(14) > 61.8 = range (avoid), < 38.2 = trend (favor)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        hhll = highest_high - lowest_low
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(atr_sum / np.log(10) / hhll)
        return chop
    
    chop_values = calculate_chop(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_values)
    
    # Calculate Williams Fractals from 12h OHLC
    bearish_fractal, bullish_fractal = calculate_williams_fractals(
        df_12h['high'].values, df_12h['low'].values
    )
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 12h EMA warmup and volume MA
    start_idx = max(54, 21)  # EMA50 needs ~50, plus 2 for fractal delay = 52; using 54 for safety; vol MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Trend filter: price relative to 12h EMA50
        bullish_bias = curr_close > ema_12h_aligned[i]
        bearish_bias = curr_close < ema_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Williams Fractal breakout + trend + volume + regime
            # Long: price breaks above bearish fractal (sell signal flipped) AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > bearish_fractal_aligned[i]) and bullish_bias and vol_spike and not_choppy
            # Short: price breaks below bullish fractal (buy signal flipped) AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < bullish_fractal_aligned[i]) and bearish_bias and vol_spike and not_choppy
            
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
            # Exit: price falls below bullish fractal (buy signal) OR loss of bullish bias OR choppy regime
            if (curr_low < bullish_fractal_aligned[i]) or (curr_close < ema_12h_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bearish fractal (sell signal) OR loss of bearish bias OR choppy regime
            if (curr_high > bearish_fractal_aligned[i]) or (curr_close > ema_12h_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "6h"
leverage = 1.0