#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H4 resistance AND 1w EMA(50) > EMA(200) (bullish weekly trend) AND 6h volume > 1.5x 20-bar avg
# - Short when price breaks below L4 support AND 1w EMA(50) < EMA(200) (bearish weekly trend) AND 6h volume > 1.5x 20-bar avg
# - Exit when price crosses back through the H3/L3 level (mean reversion to Camarilla equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivot breakout captures institutional order flow; weekly EMA filter ensures alignment with major trend
# - Volume confirmation avoids low-liquidity false breakouts
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, mean reversion exit works in ranges

name = "6h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Pre-compute 1d OHLC for Camarilla pivot calculation (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of yesterday)
    # Camarilla: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    #          H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's OHLC
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.1 * hl_range / 2
    L4 = close_1d - 1.1 * hl_range / 2
    H3 = close_1d + 1.1 * hl_range / 4
    L3 = close_1d - 1.1 * hl_range / 4
    
    # Align 1d Camarilla levels to 6h timeframe (use previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H4 resistance AND 1w bullish trend AND volume spike
            if (prices['close'].iloc[i] > H4_aligned[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L4 support AND 1w bearish trend AND volume spike
            elif (prices['close'].iloc[i] < L4_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price returns to H3/L3 level
            # Exit when price crosses back through H3 (for long) or L3 (for short)
            if position == 1:  # Long position
                exit_signal = prices['close'].iloc[i] < H3_aligned[i]  # Price crossed below H3
            else:  # Short position
                exit_signal = prices['close'].iloc[i] > L3_aligned[i]  # Price crossed above L3
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals