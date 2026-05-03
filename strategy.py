#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w trend filter and volume confirmation.
# Long when price breaks above the most recent bullish fractal in 1w uptrend with volume spike (>1.5x 20-period volume MA).
# Short when price breaks below the most recent bearish fractal in 1w downtrend with volume spike.
# Williams Fractals identify significant swing points that act as support/resistance.
# 1w EMA34 ensures higher timeframe alignment, avoiding counter-trend trades.
# Volume spike confirms institutional participation. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsFractal_1wEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams Fractals and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractals: bearish (high) at index i if high[i] is highest of i-2,i-1,i,i+1,i+2
    # bullish (low) at index i if low[i] is lowest of i-2,i-1,i,i+1,i+2
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] >= high_1w[i-1] and high_1w[i] >= high_1w[i-2] and 
            high_1w[i] >= high_1w[i+1] and high_1w[i] >= high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        if (low_1w[i] <= low_1w[i-1] and low_1w[i] <= low_1w[i-2] and 
            low_1w[i] <= low_1w[i+1] and low_1w[i] <= low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Align fractals to lower timeframe with additional delay for confirmation
    # Williams fractals need 2 extra bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Get 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above the most recent bullish fractal AND 1w uptrend AND volume spike
            # Find the most recent bullish fractal level
            recent_bullish = bullish_fractal_aligned[i]
            if not np.isnan(recent_bullish) and close_val > recent_bullish and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below the most recent bearish fractal AND 1w downtrend AND volume spike
            elif not np.isnan(bearish_fractal_aligned[i]) and close_val < bearish_fractal_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price breaks below the most recent bullish fractal
            recent_bullish = bullish_fractal_aligned[i]
            if not np.isnan(recent_bullish) and close_val < recent_bullish:
                exit_signal = True
            # Exit: 1w trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price breaks above the most recent bearish fractal
            recent_bearish = bearish_fractal_aligned[i]
            if not np.isnan(recent_bearish) and close_val > recent_bearish:
                exit_signal = True
            # Exit: 1w trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals