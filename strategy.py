#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_VolumeRegime_ATRStop_V1
Hypothesis: Daily Williams Fractal breakouts with volume confirmation (>1.5x 20-day volume MA) and weekly trend filter (price > weekly EMA34 for longs, < weekly EMA34 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.5*ATR. Designed for low trade frequency (<100 total 1d trades) to minimize fee drag and work in both bull/bear markets via regime adaptation. Uses 1w HTF for trend filter to avoid whipsaws in sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily indicators (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Fractals (5-bar: 2 left, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Additional 2-bar delay for fractal confirmation (needs 2 subsequent daily candles to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Fractal breakout conditions
        bullish_breakout = bullish_fractal_aligned[i] and price > high_1d[i]
        bearish_breakout = bearish_fractal_aligned[i] and price < low_1d[i]
        
        if position == 0:
            # Long: Bullish fractal breakout + volume + weekly uptrend filter
            if bullish_breakout and vol_ok and price > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bearish fractal breakout + volume + weekly downtrend filter
            elif bearish_breakout and vol_ok and price < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: bearish fractal or loss of volume/momentum
            elif bearish_fractal_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: bullish fractal or loss of volume/momentum
            elif bullish_fractal_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_VolumeRegime_ATRStop_V1"
timeframe = "1d"
leverage = 1.0