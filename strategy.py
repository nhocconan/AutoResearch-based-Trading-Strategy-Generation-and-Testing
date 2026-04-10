#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1w close > 1w EMA50 AND 1d volume > 1.5x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low AND 1w close < 1w EMA50 AND 1d volume > 1.5x 20-period volume SMA
# - Exit: price retreats to midpoint of Donchian channel or loss of volume confirmation
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 7-25 trades/year on 1d timeframe to stay within fee drag limits
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in bear markets

name = "1d_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Donchian channel (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Exit conditions: price retreats to midpoint or loss of volume confirmation
        exit_long = close[i] < donchian_mid[i] or not vol_confirm
        exit_short = close[i] > donchian_mid[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals