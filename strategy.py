#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams Fractal breakout with 12h trend filter and volume confirmation
    # Williams Fractals identify key swing points where price reverses. Breakouts above
    # bearish fractals or below bullish fractals with volume indicate strong momentum.
    # 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
    # This combination works in both bull and bear markets by capturing momentum shifts.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Fractals and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Fractals (5-point pattern: bar higher/lower than 2 bars each side)
    def calculate_williams_fractals(high, low):
        n = len(high)
        bearish = np.full(n, np.nan)
        bullish = np.full(n, np.nan)
        for i in range(2, n - 2):
            if (high[i] > high[i-1] and high[i] > high[i-2] and 
                high[i] > high[i+1] and high[i] > high[i+2]):
                bearish[i] = high[i]  # Bearish fractal (peak)
            if (low[i] < low[i-1] and low[i] < low[i-2] and 
                low[i] < low[i+1] and low[i] < low[i+2]):
                bullish[i] = low[i]   # Bullish fractal (trough)
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_12h, low_12h)
    
    # Williams fractals need 2 extra 12h bars for confirmation (bar forms, needs 2 more to close)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 12h EMA50 trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above bearish fractal resistance with volume + price above 12h EMA50 (uptrend)
            if close[i] > bearish_fractal_aligned[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below bullish fractal support with volume + price below 12h EMA50 (downtrend)
            elif close[i] < bullish_fractal_aligned[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite fractal level or trend reversal vs 12h EMA50
            if position == 1:
                if close[i] < bullish_fractal_aligned[i] or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bearish_fractal_aligned[i] or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hEMA50_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0