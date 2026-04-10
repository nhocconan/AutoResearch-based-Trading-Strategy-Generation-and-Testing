#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# - Long: Price breaks above Donchian(20) high + price > 1d EMA50 (uptrend) + 1d volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) low + price < 1d EMA50 (downtrend) + 1d volume > 1.5x 20-period MA
# - Exit: Price returns to Donchian(20) midpoint or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Uses 1d HTF for trend and volume to filter false breakouts in choppy markets
# - Donchian breakout captures momentum; 1d EMA/volume ensures alignment with higher timeframe momentum
# - Works in bull/bear: breakouts in trends with institutional participation

name = "12h_1d_donchian_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) for 12h
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for Donchian20 and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get aligned 1d data for current 12h bar (completed 1d bar)
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume spike condition: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + price above 1d EMA50 + volume spike
            if (close_price > highest_high[i] and close_price > ema_50_current and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) low + price below 1d EMA50 + volume spike
            elif (close_price < lowest_low[i] and close_price < ema_50_current and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian(20) midpoint or opposite signal
            if position == 1 and close_price <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals