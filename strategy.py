#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# - Long when BB Width(20) < 20th percentile (squeeze) AND price breaks above upper BB AND 1d EMA(50) > EMA(200) AND 6h volume > 2.0x 20-bar avg
# - Short when BB Width(20) < 20th percentile (squeeze) AND price breaks below lower BB AND 1d EMA(50) < EMA(200) AND 6h volume > 2.0x 20-bar avg
# - Exit when price returns to middle BB (20-period SMA)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Bollinger squeeze identifies low volatility contraction before expansion breakout
# - 1d EMA filter ensures alignment with daily trend to avoid counter-trend trades
# - Volume confirmation (2.0x avg) ensures institutional participation
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: squeeze breakouts occur in all regimes, trend filter prevents wrong-direction trades

name = "6h_1d_bb_squeeze_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMA trend to 6h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute Bollinger Bands (20, 2) on 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Middle Band = 20-period SMA
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Standard Deviation of close
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    
    # Upper and Lower Bands
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    # Bollinger Band Width = (Upper - Lower) / Middle
    bb_width = (upper_bb - lower_bb) / sma_20
    # Handle division by zero (when sma_20 == 0)
    bb_width = np.where(sma_20 == 0, 0, bb_width)
    
    # BB Width percentile (20-period lookback for squeeze definition)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Squeeze condition: BB Width < 20th percentile
    squeeze = bb_width_percentile < 0.20
    
    # Breakout conditions
    breakout_up = close > upper_bb  # Price breaks above upper BB
    breakout_down = close < lower_bb  # Price breaks below lower BB
    
    # Return to middle band (exit condition)
    return_to_middle = np.abs(close - sma_20) < (0.1 * std_20)  # Within 10% of std dev from middle
    
    # Pre-compute 6h volume confirmation: > 2.0x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(squeeze[i]) or np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(return_to_middle[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new squeeze breakout entries
            # Long when squeeze AND breakout up AND 1d bullish trend AND volume spike
            if (squeeze[i] and 
                breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when squeeze AND breakout down AND 1d bearish trend AND volume spike
            elif (squeeze[i] and 
                  breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to middle BB
            # Exit when price returns to middle BB
            exit_signal = return_to_middle[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals