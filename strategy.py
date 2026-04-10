#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation
# - Long when price breaks above Donchian(20) high AND weekly pivot > prior weekly pivot (bullish week) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND weekly pivot < prior weekly pivot (bearish week) AND volume > 1.5x 20-bar avg
# - Exit when price crosses Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian captures structural breaks; weekly pivot ensures alignment with higher timeframe bias
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, midpoint exit in ranges

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian(20) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute weekly pivot direction: bullish if current pivot > prior weekly pivot
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    typical_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    pivot_1w = typical_1w.values
    pivot_bullish = np.roll(pivot_1w, 1) < pivot_1w  # current > prior
    pivot_bearish = np.roll(pivot_1w, 1) > pivot_1w  # current < prior
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Align HTF indicators to 6h timeframe
    pivot_bullish_aligned = align_htf_to_ltf(prices, df_1w, pivot_bullish)
    pivot_bearish_aligned = align_htf_to_ltf(prices, df_1w, pivot_bearish)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute RSI(14) for exit filter (optional, but helps in choppy markets)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50, rsi)
    rsi_mid = (rsi > 40) & (rsi < 60)  # RSI near midpoint for exit
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_bullish_aligned[i]) or np.isnan(pivot_bearish_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND weekly bullish pivot AND volume spike
            if (prices['close'].iloc[i] > donchian_high[i] and 
                pivot_bullish_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND weekly bearish pivot AND volume spike
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  pivot_bearish_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses Donchian midpoint OR RSI returns to midpoint (double confirmation)
            exit_signal = (
                (position == 1 and prices['close'].iloc[i] < donchian_mid[i]) or
                (position == -1 and prices['close'].iloc[i] > donchian_mid[i]) or
                rsi_mid[i]
            )
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals