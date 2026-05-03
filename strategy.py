#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high with volume > 1.5x 20-bar average and close > 1d EMA50 (uptrend)
# Short when price breaks below 20-period Donchian low with volume > 1.5x 20-bar average and close < 1d EMA50 (downtrend)
# Exit when price crosses 1d EMA50 (trend reversal) or opposite Donchian breakout occurs
# Donchian channels provide clear structure, EMA50 filters for higher-timeframe trend, volume confirms breakout strength
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.
# Works in both bull and bear markets: in bull, longs catch uptrends; in bear, shorts catch downtrends; EMA filter avoids counter-trend trades.

name = "12h_Donchian20_Volume_1dEMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values  # shift(1) to avoid look-ahead
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 1  # EMA50(1d) + Donchian(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume spike and close > 1d EMA50 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike and close < 1d EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 1d EMA50 (trend reversal) or breaks below Donchian low (contrarian signal)
            if (close[i] < ema_50_aligned[i] or 
                close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA50 (trend reversal) or breaks above Donchian high (contrarian signal)
            if (close[i] > ema_50_aligned[i] or 
                close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals