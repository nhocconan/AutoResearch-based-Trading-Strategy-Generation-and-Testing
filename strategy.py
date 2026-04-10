#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above H3 level AND 4h EMA(20) > EMA(50) (bullish trend) AND 1h volume > 1.8x 20-bar avg
# - Short when price breaks below L3 level AND 4h EMA(20) < EMA(50) (bearish trend) AND 1h volume > 1.8x 20-bar avg
# - Exit when price returns to pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Camarilla levels provide intraday support/resistance; 4h EMA filter ensures alignment with intermediate trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: breakout in trends, pivot reversion in ranges

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA trend filter: EMA(20) vs EMA(50)
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish = ema_20 > ema_50
    ema_bearish = ema_20 < ema_50
    
    # Align 4h EMA trend to 1h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish)
    
    # Pre-compute 1h volume confirmation: > 1.8x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * volume_20_avg)
    
    # Pre-compute Camarilla pivot levels (using previous day's OHLC)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Calculate daily pivot from previous day's data
    # We'll use rolling window of 24h (96 bars for 15m, but for 1h we need 24 bars)
    # Since we're on 1h timeframe, previous day = 24 bars ago
    prev_high = pd.Series(high).shift(24).rolling(window=24, min_periods=24).max().values
    prev_low = pd.Series(low).shift(24).rolling(window=24, min_periods=24).min().values
    prev_close = pd.Series(close).shift(24).rolling(window=24, min_periods=24).last().values
    
    # Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # H3 and L3 levels (most significant for breakouts)
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    
    # Handle division by zero and invalid values
    h3 = np.where((range_val == 0) | np.isnan(range_val), pivot, h3)
    l3 = np.where((range_val == 0) | np.isnan(range_val), pivot, l3)
    pivot = np.where(np.isnan(pivot), close, pivot)
    
    # Breakout conditions
    breakout_long = (close > h3) & ~np.isnan(h3)
    breakout_short = (close < l3) & ~np.isnan(l3)
    
    # Exit condition: price returns to pivot point (within 0.1% of pivot)
    pivot_exit = np.abs(close - pivot) < (0.001 * pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(breakout_long[i]) or np.isnan(breakout_short[i]) or
            np.isnan(pivot_exit[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND volume spike
            if (breakout_long[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND volume spike
            elif (breakout_short[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point
            # Exit when price returns to pivot (mean reversion)
            exit_signal = pivot_exit[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals