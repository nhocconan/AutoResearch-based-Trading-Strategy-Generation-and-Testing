#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with volume confirmation and 1d trend filter.
Long when price breaks above bullish fractal with volume > 1.4x average and daily close > EMA34.
Short when price breaks below bearish fractal with volume > 1.4x average and daily close < EMA34.
Exit when price returns to the opposite fractal level or volume drops below average.
Williams Fractals provide swing high/low structure, volume confirms breakout strength,
and EMA34 filters for trend alignment to avoid counter-trend trades in chop.
Target: 12-37 trades/year for low fee drag and robust performance in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for fractals, EMA34, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Fractals (requires 5-bar window: 2 left, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Calculate daily EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Apply 2-bar delay for fractal confirmation (needs 2 future bars to confirm)
    bearish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (12h close and volume aligned to daily)
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: price breaks above bullish fractal, volume surge, daily close > EMA34 (uptrend)
            if (price_close > bullish_fractal_confirmed[i] and 
                vol_1d_current > 1.4 * vol_ma_20_aligned[i] and
                close_1d[-1] > ema_34[-1]):  # Use latest daily close vs EMA
                # Actually need to check current daily EMA condition
                # Get the daily index corresponding to current 12h bar
                # Since we aligned, we can use the aligned EMA value
                if price_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: price breaks below bearish fractal, volume surge, daily close < EMA34 (downtrend)
            elif (price_close < bearish_fractal_confirmed[i] and 
                  vol_1d_current > 1.4 * vol_ma_20_aligned[i] and
                  close_1d[-1] < ema_34[-1]):
                if price_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit: price returns to opposite fractal or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= bearish fractal or volume < average
                if (price_close <= bearish_fractal_confirmed[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= bullish fractal or volume < average
                if (price_close >= bullish_fractal_confirmed[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_Volume1.4x_EMA34"
timeframe = "12h"
leverage = 1.0