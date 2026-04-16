#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above recent bearish fractal (swing high) + 1w EMA34 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below recent bullish fractal (swing low) + 1w EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams Fractals provide natural support/resistance levels that work in ranging and trending markets.
# 1w EMA34 provides strong multi-week trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-25 trades/year on 1d timeframe to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicator: EMA34 ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Williams Fractals (5-bar) ===
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    # Using rolling window with min_periods=5 to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Bearish fractal (swing high): current high is max in window
    bearish_fractal = high_series.rolling(window=5, center=True, min_periods=5).max().values
    # Bullish fractal (swing low): current low is min in window
    bullish_fractal = low_series.rolling(window=5, center=True, min_periods=5).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20, 5) + 5  # EMA34 + Donchian(20) + Fractals(5) + buffer
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal[i]) or np.isnan(bullish_fractal[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above recent bearish fractal (swing high)
        # 2. 1w EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > bearish_fractal[i]) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below recent bullish fractal (swing low)
        # 2. 1w EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < bullish_fractal[i]) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0