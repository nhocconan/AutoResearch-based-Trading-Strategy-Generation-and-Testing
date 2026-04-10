#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w EMA50 trend filter
# - Long when price breaks above 1d Donchian upper channel AND 1w volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA50
# - Short when price breaks below 1d Donchian lower channel AND 1w volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA50
# - Exit: price retreats to the opposite Donchian channel (middle) or loss of volume confirmation
# - Position sizing: 0.30 discrete level to balance return and drawdown
# - Target: 7-25 trades/year on 1d timeframe to stay within fee drag limits
# - Uses Donchian channels for structure, 1w for trend and volume confirmation

name = "1d_donchian_breakout_1w_volume_trend_v1"
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
    
    # Calculate 1d Donchian channels (20-period)
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    # Middle channel = (upper + lower) / 2
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1w volume SMA for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1w volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1w[i // (7*24*4)] > 1.5 * volume_sma_20_1w_aligned[i] if i // (7*24*4) < len(volume_1w) else False
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper channel
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower channel
        
        # Exit conditions: price retreats to middle channel or loss of volume confirmation
        exit_long = close[i] < donchian_middle[i] or not vol_confirm
        exit_short = close[i] > donchian_middle[i] or not vol_confirm
        
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