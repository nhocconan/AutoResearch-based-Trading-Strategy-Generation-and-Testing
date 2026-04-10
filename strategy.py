#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w close > 1w EMA50 AND 1d volume > 1.5x 20-day volume SMA
# - Short when price breaks below 20-day low AND 1w close < 1w EMA50 AND 1d volume > 1.5x 20-day volume SMA
# - Exit: price retreats to 10-day EMA (adaptive stoploss)
# - Position sizing: 0.30 discrete level
# - Target: 15-25 trades/year on 1d timeframe to stay within fee drag limits
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in bear markets

name = "1d_donchian_breakout_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_rolling_max[i-1]) or np.isnan(low_rolling_min[i-1]) or
            np.isnan(ema_10[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout signals (using previous bar's channels)
        breakout_up = close[i] > high_rolling_max[i-1]
        breakout_down = close[i] < low_rolling_min[i-1]
        
        # Exit condition: price retreats to 10-period EMA
        exit_long = close[i] < ema_10[i]
        exit_short = close[i] > ema_10[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.30
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals