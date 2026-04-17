#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above latest bullish fractal AND volume > 1.5x 20-period average AND price > 1w EMA34.
Short when price breaks below latest bearish fractal AND volume > 1.5x 20-period average AND price < 1w EMA34.
Exit when price crosses the 1w EMA34 in opposite direction.
Williams Fractals identify key swing points, 1w EMA34 filters for higher timeframe trend,
volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
by trading with the 1w trend while using fractals for precise entry/exit.
Targets 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w timeframe
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average (20-period) on 1d
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Fractals on 1d timeframe (need 5 bars: 2 left, center, 2 right)
    # Using high and low arrays from 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    
    # Align all indicators to 1d timeframe
    # Note: Williams fractals need additional 2-bar delay for confirmation
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma)  # 1d to 1d alignment (no change)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators (Williams fractals + EMA34)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_34 = ema_34_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        bearish_fract = bearish_fractal_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal AND volume > 1.5x avg AND price > 1w EMA34 (bullish trend)
            if high_price > bullish_fract and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal AND volume > 1.5x avg AND price < 1w EMA34 (bearish trend)
            elif low_price < bearish_fract and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1w EMA34
            if price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1w EMA34
            if price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0