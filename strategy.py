#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams Fractals for swing high/low identification combined with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above the most recent bullish fractal (swing high) AND 1w EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below the most recent bearish fractal (swing low) AND 1w EMA34 is falling AND volume > 2.0x 20-period average.
Exit when price retouches the most recent opposite fractal or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 6h timeframe to target 12-25 trades/year per symbol (50-100 total over 4 years).
Williams Fractals provide objective swing points that work in both trending and ranging markets, while 1w EMA34 filters counter-trend moves and volume confirmation avoids false breakouts.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 1d Williams Fractals (swing points)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Compute Williams Fractals: need 5 bars (2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # EMA slope (rising/falling)
    ema_slope = np.zeros_like(ema_1w_34_aligned)
    ema_slope[1:] = ema_1w_34_aligned[1:] - ema_1w_34_aligned[:-1]
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 10, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_1w_34_aligned[i]) or np.isnan(ema_slope[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above most recent bullish fractal AND 1w EMA34 rising AND volume spike
            if (not np.isnan(bull_fractal) and price > bull_fractal and 
                ema_slope_val > 0 and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below most recent bearish fractal AND 1w EMA34 falling AND volume spike
            elif (not np.isnan(bear_fractal) and price < bear_fractal and 
                  ema_slope_val < 0 and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches the most recent opposite fractal
            if position == 1 and not np.isnan(bear_fractal) and price <= bear_fractal:
                exit_signal = True
            elif position == -1 and not np.isnan(bull_fractal) and price >= bull_fractal:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractals_1wEMA34_Trend_VolumeConfirmation_ATRStop"
timeframe = "6h"
leverage = 1.0