#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w trend filter and volume confirmation
# Williams Fractals identify swing highs/lows - breakouts above/below these levels
# with volume and weekly trend alignment capture strong momentum moves.
# Weekly trend filter avoids counter-trend trades. Target: 15-30 trades/year (60-120 total).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams Fractals need 2 extra daily bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Weekly EMA trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-period average)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above bullish fractal (swing high) with volume + above weekly EMA
            if (bullish_fractal_aligned[i] > 0 and  # fractal level is active
                price > bullish_fractal_aligned[i] and
                volume_confirm and
                price > weekly_ema):
                position = 1
                signals[i] = position_size
            # Short: break below bearish fractal (swing low) with volume + below weekly EMA
            elif (bearish_fractal_aligned[i] > 0 and  # fractal level is active
                  price < bearish_fractal_aligned[i] and
                  volume_confirm and
                  price < weekly_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below bullish fractal or reaches bearish fractal
            if (price < bullish_fractal_aligned[i] or
                price < bearish_fractal_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns above bearish fractal or reaches bullish fractal
            if (price > bearish_fractal_aligned[i] or
                price > bullish_fractal_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Williams_Fractal_Breakout_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0