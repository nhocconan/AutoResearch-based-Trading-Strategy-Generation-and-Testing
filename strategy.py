#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Camarilla levels from 1h: breakout above H3 = long, below L3 = short
# - 4h EMA(50) > EMA(200) for long bias, < for short bias to avoid counter-trend trades
# - Volume confirmation: current 1h volume > 1.5x 20-period average
# - Designed for 1h timeframe: targets 15-37 trades/year to avoid fee drag
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Works in bull/bear markets: 4h EMA filter ensures we trade with higher timeframe trend

name = "1h_4h_camarilla_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) and EMA(200) for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Pre-compute 1h Camarilla levels (based on previous bar's OHLC)
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    open_1h = prices['open'].values
    
    # Camarilla levels use previous bar's OHLC
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close = np.roll(close_1h, 1)
    prev_open = np.roll(open_1h, 1)
    
    # Handle first bar
    prev_high[0] = high_1h[0]
    prev_low[0] = low_1h[0]
    prev_close[0] = close_1h[0]
    prev_open[0] = open_1h[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    h4 = pivot + (range_val * 1.1 / 2)
    l4 = pivot - (range_val * 1.1 / 2)
    
    # Pre-compute 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters Camarilla range (below H3) or trend reverses
            if close_1h[i] < h3[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price re-enters Camarilla range (above L3) or trend reverses
            if close_1h[i] > l3[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above H3 with bullish 4h trend
                if close_1h[i] > h3[i] and ema_50_aligned[i] > ema_200_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Breakout short: price closes below L3 with bearish 4h trend
                elif close_1h[i] < l3[i] and ema_50_aligned[i] < ema_200_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals