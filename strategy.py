#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation.
# Williams Fractal identifies potential reversal points: bearish fractal (sell signal) when a candle has lower highs on both sides,
# bullish fractal (buy signal) when a candle has higher lows on both sides.
# Breakout logic: go long when price breaks above a bullish fractal level in an uptrend (1d EMA34 rising),
# go short when price breaks below a bearish fractal level in a downtrend (1d EMA34 falling).
# Volume > 1.3x 20-period average confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) by requiring fractal alignment + trend + volume.
# Works in bull/bear: EMA34 trend filter avoids whipsaws in ranging markets; fractals provide natural support/resistance.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n1d = len(high_1d)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    
    # Williams Fractal: need 2 candles on each side (5 total)
    for i in range(2, n1d - 2):
        # Bullish fractal: lowest low in the middle
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
        # Bearish fractal: highest high in the middle
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
    
    # Align 1d indicators to 6h timeframe with extra delay for fractals (need confirmation)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient warmup
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: EMA34 slope (rising/falling)
        ema_now = ema_34_aligned[i]
        ema_prev = ema_34_aligned[i-1] if i > 0 else ema_now
        ema_rising = ema_now > ema_prev
        ema_falling = ema_now < ema_prev
        
        # Fractal levels (only valid when fractal is present)
        bullish_level = None
        bearish_level = None
        if bullish_fractal_aligned[i] > 0.5:  # Fractal present
            # Find the actual low value from 2 days ago (due to 2-bar delay in alignment)
            idx_1d = i // 4  # Approximate 6h to 1d index (4x 6h bars per day)
            if idx_1d >= 2 and idx_1d < len(low_1d) - 2:
                bullish_level = low_1d[idx_1d]
        if bearish_fractal_aligned[i] > 0.5:  # Fractal present
            idx_1d = i // 4
            if idx_1d >= 2 and idx_1d < len(high_1d) - 2:
                bearish_level = high_1d[idx_1d]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above bullish fractal level in uptrend
                if bullish_level is not None and price > bullish_level and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below bearish fractal level in downtrend
                elif bearish_level is not None and price < bearish_level and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below bullish fractal level (failed breakout) or trend turns down
                if bullish_level is not None and price < bullish_level:
                    exit_signal = True
                elif ema_falling:  # Trend turned down
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above bearish fractal level (failed breakdown) or trend turns up
                if bearish_level is not None and price > bearish_level:
                    exit_signal = True
                elif ema_rising:  # Trend turned up
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0