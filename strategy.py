#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Fractals for swing high/low detection with 12h HMA trend filter and volume confirmation
# Long when price breaks above prior 1d bullish fractal (swing high) in uptrend (12h close > 12h HMA21) with volume spike (>2.0x average)
# Short when price breaks below prior 1d bearish fractal (swing low) in downtrend (12h close < 12h HMA21) with volume spike
# Williams Fractals require 2-bar confirmation after the center bar, so we use additional_delay_bars=2 when aligning
# Designed for low trade frequency (~15-25/year on 4h) to minimize fee drag while capturing continuation of swing points
# Stoploss at 1.5 * ATR below/above entry, no take profit (let winners run until reversal signal)
# Works in bull markets via buying swing high breaks and in bear markets via selling swing low breaks

name = "4h_1dWilliamsFractals_Breakout_12hHMA21_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: 5-bar pattern (high/low surrounded by 2 lower highs/2 higher lows)
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)  # swing high
    bullish_fractal = np.full(n_1d, np.nan)  # swing low
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (swing high): high[i] is highest of 5 bars
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (swing low): low[i] is lowest of 5 bars
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align 1d Williams Fractals to 4h timeframe with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 12h HMA(21) for trend filter: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        wma = np.convolve(values, weights, mode='valid') / weights.sum()
        # Pad with NaN for alignment
        return np.concatenate([np.full(window-1, np.nan), wma])
    
    close_12h = df_12h['close'].values
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    wma_half = calculate_wma(close_12h, half_len)
    wma_full = calculate_wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = calculate_wma(raw_hma, sqrt_len)
    
    # Align 12h HMA to 4h timeframe
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average (moderate to balance trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_bear_fractal = bearish_fractal_aligned[i]  # swing high
        curr_bull_fractal = bullish_fractal_aligned[i]  # swing low
        curr_hma = hma_21_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike and not np.isnan(curr_hma):
                # Bullish entry: price breaks above prior 1d bullish fractal (swing low) with 12h uptrend
                if (not np.isnan(curr_bull_fractal) and 
                    curr_close > curr_bull_fractal and 
                    curr_close > curr_hma):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below prior 1d bearish fractal (swing high) with 12h downtrend
                elif (not np.isnan(curr_bear_fractal) and 
                      curr_close < curr_bear_fractal and 
                      curr_close < curr_hma):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 1.5 * ATR below entry price OR price breaks below prior 1d bullish fractal (swing low break)
            if curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif (not np.isnan(curr_bull_fractal) and curr_close < curr_bull_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 1.5 * ATR above entry price OR price breaks above prior 1d bearish fractal (swing high break)
            if curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif (not np.isnan(curr_bear_fractal) and curr_close > curr_bear_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals