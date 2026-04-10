#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# - Long when BB width < 20th percentile (squeeze) AND price breaks above upper band 
#   AND 1w EMA(50) > EMA(200) (bullish trend) AND 6h volume > 2.0x 20-bar avg
# - Short when BB width < 20th percentile (squeeze) AND price breaks below lower band
#   AND 1w EMA(50) < EMA(200) (bearish trend) AND 6h volume > 2.0x 20-bar avg
# - Exit when price returns to middle band (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Bollinger squeeze captures low volatility pre-breakout; 1w EMA filter ensures alignment with long-term trend
# - Volume confirmation avoids low-liquidity false breakouts
# - Works in both bull and bear markets: breakouts in expansion phases, mean reversion in ranges
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_bb_squeeze_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(50) vs EMA(200)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1w EMA trend to 6h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish)
    
    # Pre-compute Bollinger Bands (20, 2.0) on 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Middle band = SMA(20)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Standard deviation
    std_dev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    
    # Upper and lower bands
    upper_band = sma_20 + (2.0 * std_dev)
    lower_band = sma_20 - (2.0 * std_dev)
    
    # Bollinger Band Width = (Upper - Lower) / Middle
    bb_width = (upper_band - lower_band) / sma_20
    # Handle division by zero (when sma_20 == 0)
    bb_width = np.where(sma_20 == 0, 0, bb_width)
    
    # BB width percentile (20-period lookback for squeeze detection)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Squeeze condition: BB width < 20th percentile
    squeeze = bb_width_percentile < 0.20
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
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
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new squeeze breakout entries
            # Long when squeeze AND breakout up AND 1w bullish trend AND volume spike
            if (squeeze[i] and 
                breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when squeeze AND breakout down AND 1w bearish trend AND volume spike
            elif (squeeze[i] and 
                  breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to middle band (mean reversion)
            # Exit when price returns to middle band (SMA20)
            exit_signal = (position == 1 and close[i] <= sma_20[i]) or \
                          (position == -1 and close[i] >= sma_20[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals