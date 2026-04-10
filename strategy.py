#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume regime filter and 1w EMA50 trend filter
# - Long when price breaks above 12h Donchian upper channel AND 1w EMA50 rising (bullish trend) AND 12h volume > 1.8x 20-period volume SMA (strong volume)
# - Short when price breaks below 12h Donchian lower channel AND 1w EMA50 falling (bearish trend) AND 12h volume > 1.8x 20-period volume SMA
# - Exit: opposite Donchian breakout or volume drops below 1.2x 20-period volume SMA
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-30 trades/year on 12h timeframe to stay within fee drag limits
# - Uses 1w EMA50 slope for trend filter (more stable than price vs EMA)
# - Volume regime filter avoids low-volume breakouts that often fail

name = "12h_1w_ema50_vol_regime_donchian_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w EMA50 slope (rising/falling trend)
    ema_50_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Calculate 12h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.8x 20-period volume SMA (strong volume)
        vol_strong = volume[i] > 1.8 * volume_sma_20[i]
        # Volume exit: drop below 1.2x SMA (weak volume)
        vol_weak = volume[i] < 1.2 * volume_sma_20[i]
        
        # Trend filter: 1w EMA50 slope
        trend_bullish = ema_50_rising[i]
        trend_bearish = ema_50_falling[i]
        
        # Donchian breakout signals (using previous bar's channel)
        breakout_up = close[i] > donchian_upper[i-1]
        breakout_down = close[i] < donchian_lower[i-1]
        
        # Exit conditions
        exit_long = breakout_down or vol_weak
        exit_short = breakout_up or vol_weak
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_strong:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_strong:
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