#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1-day trend filter and volume confirmation.
# Williams Fractals identify swing highs/lows; breakouts above recent fractal highs
# or below recent fractal lows with volume and aligned daily trend indicate strong momentum.
# Designed for 6h timeframe to capture medium-term breakouts with low frequency and high win rate.
# Entry: Long when close > recent bearish fractal high and daily EMA34 > daily EMA89 and volume spike.
#        Short when close < recent bullish fractal low and daily EMA34 < daily EMA89 and volume spike.
# Exit: Opposite fractal level touch or daily EMA crossover.
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
name = "6h_WilliamsFractal_EMA_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align EMAs to 6h timeframe (waits for completed daily candle)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89)
    
    # Williams Fractals (5-bar pattern: high[low-low-2,low-1,low,low+1,low+2])
    # Bearish fractal: high[i] is highest among [i-2,i-1,i,i+1,i+2]
    # Bullish fractal: low[i] is lowest among [i-2,i-1,i,i+1,i+2]
    n1d = len(high)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Align fractals to 6h with 2-day delay for confirmation (fractals need 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above recent bearish fractal high with bullish daily trend and volume
            if (close[i] > bearish_fractal_aligned[i] and 
                ema_34_aligned[i] > ema_89_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below recent bullish fractal low with bearish daily trend and volume
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches bullish fractal low or daily EMA turns bearish
            if (close[i] < bullish_fractal_aligned[i]) or (ema_34_aligned[i] < ema_89_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches bearish fractal high or daily EMA turns bullish
            if (close[i] > bearish_fractal_aligned[i]) or (ema_34_aligned[i] > ema_89_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals