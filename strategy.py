#!/usr/bin/env python3
# 1d_williams_fractal_breakout_v1
# Hypothesis: Daily strategy using Williams fractals for breakout detection with 1w HTF trend filter.
# Williams fractals identify potential reversal/continuation points. Long when bullish fractal forms
# above price with 1w uptrend, short when bearish fractal forms below price with 1w downtrend.
# Volume confirmation filters false signals. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 30-100 total trades over 4 years by requiring fractal breakout + volume + 1w trend alignment.
# Primary timeframe: 1d, HTF: 1w for trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "1d_williams_fractal_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1w HTF data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # 1w EMA34 aligned to 1d timeframe (completed 1w bar only)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Williams fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices[['high','low']], bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices[['high','low']], bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below bearish fractal level OR 1w trend turns bearish
            if close[i] < bearish_fractal_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bullish fractal level OR 1w trend turns bullish
            if close[i] > bullish_fractal_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long breakout: bullish fractal forms above price with 1w uptrend
                if (bullish_fractal_aligned[i] > 0 and  # bullish fractal present
                    close[i] > bullish_fractal_aligned[i] and  # price above fractal level
                    close[i] > ema_34_1w_aligned[i]):  # 1w uptrend
                    position = 1
                    signals[i] = 0.25
                # Short breakout: bearish fractal forms below price with 1w downtrend
                elif (bearish_fractal_aligned[i] > 0 and  # bearish fractal present
                      close[i] < bearish_fractal_aligned[i] and  # price below fractal level
                      close[i] < ema_34_1w_aligned[i]):  # 1w downtrend
                    position = -1
                    signals[i] = -0.25
    
    return signals