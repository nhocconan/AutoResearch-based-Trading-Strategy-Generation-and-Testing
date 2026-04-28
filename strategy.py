#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Williams Fractal breaks with 12h EMA50 trend filter and volume spike confirmation.
# Enter long on bullish fractal break above recent high with 12h EMA50 uptrend and volume > 2x 20-bar average.
# Enter short on bearish fractal break below recent low with 12h EMA50 downtrend and volume > 2x 20-bar average.
# Exit when price retraces to the 12h EMA50 (dynamic stop/reversal).
# Williams Fractals provide high-probability reversal points; 12h EMA50 ensures higher timeframe alignment;
# volume confirmation filters weak breakouts. Designed for 6h timeframe to capture multi-day swings in both bull and bear markets.

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stop management (optional, not used in signals)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for EMA50 trend filter and fractal calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Williams Fractals on 12h data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(low_12h), np.nan)
    
    # Need at least 5 points for fractal (center -2 to +2)
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] < high_12h[i-1] and 
            high_12h[i-1] > high_12h[i-3] and 
            high_12h[i-1] > high_12h[i+1]):
            bearish_fractal[i-1] = high_12h[i-1]  # Value at the fractal point
        
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] > low_12h[i-1] and 
            low_12h[i-1] < low_12h[i-3] and 
            low_12h[i-1] < low_12h[i+1]):
            bullish_fractal[i-1] = low_12h[i-1]  # Value at the fractal point
    
    # Align fractals to 6h with additional delay for confirmation (fractals need 2 extra bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish fractal break above recent high, EMA50 up, volume confirm
            # Bullish fractal value acts as support level
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                price > bullish_fractal_aligned[i] and 
                ema_trend_up and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish fractal break below recent low, EMA50 down, volume confirm
            # Bearish fractal value acts as resistance level
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  price < bearish_fractal_aligned[i] and 
                  ema_trend_down and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit when price crosses below 12h EMA50
            if price < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit when price crosses above 12h EMA50
            if price > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals