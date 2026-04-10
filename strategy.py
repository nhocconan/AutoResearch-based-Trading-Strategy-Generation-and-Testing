#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# - Long: Price breaks above Donchian(20) upper band + 1w EMA(20) > EMA(50) (uptrend) + 1w volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) lower band + 1w EMA(20) < EMA(50) (downtrend) + 1w volume > 1.5x 20-period MA
# - Exit: Price crosses Donchian(20) midline (10-period average) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Donchian breakouts capture strong momentum moves; 1w EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation filters out weak breakouts, reducing false signals

name = "12h_1w_donchian_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian(20) channels for 12h
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1w EMA(20) and EMA(50) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for Donchian20 and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get aligned 1w data for current 12h bar (completed 1w bar)
        ema_20_current = ema_20_aligned[i]
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Trend condition: EMA(20) > EMA(50) for uptrend, EMA(20) < EMA(50) for downtrend
        uptrend = ema_20_current > ema_50_current
        downtrend = ema_20_current < ema_50_current
        
        # Volume spike condition: current 1w volume > 1.5x 20-period MA
        volume_spike = volume_1w_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + uptrend + volume spike
            if (close_price > donchian_upper[i] and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + downtrend + volume spike
            elif (close_price < donchian_lower[i] and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Price crosses Donchian middle or opposite signal
            if position == 1:
                if close_price < donchian_middle[i]:  # Exit long when price crosses below middle
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > donchian_middle[i]:  # Exit short when price crosses above middle
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals