# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# 6h Williams Fractal + Volume Confirmation + 12h Trend Filter
# Williams Fractals identify potential reversal points at support/resistance.
# In trending markets (12h EMA50), we trade breakouts in the trend direction.
# In ranging markets, we fade extreme fractal touches with volume confirmation.
# This adapts to both bull and bear regimes by using the 12h trend as regime filter.
# Target: 50-150 total trades over 4 years = 12-37/year.

#!/usr/bin/env python3
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Williams Fractals (requires 2-bar confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals: need 2 extra bars for confirmation
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 10)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Fractal signals: True if fractal formed at this bar
        bearish_fractal_signal = bearish_fractal_aligned[i]
        bullish_fractal_signal = bullish_fractal_aligned[i]
        
        # Entry conditions
        # Long: bullish fractal at support in uptrend OR oversold bounce in downtrend with volume
        long_entry = (
            (bullish_fractal_signal and uptrend and volume_confirm[i]) or
            (bullish_fractal_signal and not uptrend and volume_confirm[i] and close[i] < ema_50_12h_aligned[i] * 0.98)
        )
        
        # Short: bearish fractal at resistance in downtrend OR overbought rejection in uptrend with volume
        short_entry = (
            (bearish_fractal_signal and downtrend and volume_confirm[i]) or
            (bearish_fractal_signal and not downtrend and volume_confirm[i] and close[i] > ema_50_12h_aligned[i] * 1.02)
        )
        
        # Exit conditions: opposite fractal or trend reversal
        if position == 1:
            exit_condition = bearish_fractal_signal or not uptrend
        elif position == -1:
            exit_condition = bullish_fractal_signal or not downtrend
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_VolumeConfirm_12hTrend"
timeframe = "6h"
leverage = 1.0