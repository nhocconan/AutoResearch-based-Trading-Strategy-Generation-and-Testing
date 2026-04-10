#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation
# - Primary: 12h price breaks above/below 20-period Donchian channel for entry
# - HTF: 1w close above/below 50-period EMA for trend alignment (avoids counter-trend trades)
# - Volume: 12h volume > 1.3x 20-period MA for confirmation (avoids low-volume breakouts)
# - Long: Price > Donchian Upper + 1w EMA50 up + volume confirmation
# - Short: Price < Donchian Lower + 1w EMA50 down + volume confirmation
# - Exit: Price crosses Donchian midpoint (mean reversion) or trend flips
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Donchian captures breakouts, 1w EMA filters regime, volume avoids false signals
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1w_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need enough data for 50-period EMA
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    
    # Calculate 12h Donchian Channel (20-period)
    donchian_upper = np.full(len(close_12h), np.nan)
    donchian_lower = np.full(len(close_12h), np.nan)
    donchian_middle = np.full(len(close_12h), np.nan)
    
    for i in range(19, len(close_12h)):
        if not (np.isnan(high_12h[i-19:i+1]).any() or np.isnan(low_12h[i-19:i+1]).any()):
            donchian_upper[i] = np.max(high_12h[i-19:i+1])
            donchian_lower[i] = np.min(low_12h[i-19:i+1])
            donchian_middle[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Calculate 1w EMA (50-period)
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        # Simple SMA for first value
        ema_50_1w[49] = np.mean(close_1w[:50])
        # EMA calculation
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1w)):
            if not np.isnan(close_1w[i]) and not np.isnan(ema_50_1w[i-1]):
                ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(19, len(volume_12h)):
        if not np.isnan(volume_12h[i-19:i+1]).any():
            volume_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align all HTF/LTF indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, prices, donchian_middle)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, prices, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period MA
        volume_confirm = volume_12h[i] > 1.3 * volume_ma_20_12h_aligned[i]
        
        # Trend filter: 1w EMA50 direction
        ema_trend_up = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False  # Current 1w trend
        ema_trend_down = close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False
        
        # Get current 1w EMA50 value (aligned)
        current_ema_50 = ema_50_1w_aligned[i]
        price_1w = close_1w[-1] if len(close_1w) > 0 else np.nan  # Current 1w close
        
        # Determine 1w trend based on aligned values
        if not np.isnan(current_ema_50) and not np.isnan(price_1w):
            ema_trend_up = price_1w > current_ema_50
            ema_trend_down = price_1w < current_ema_50
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Donchian Upper + 1w EMA50 up + volume confirmation
            if close_12h[i] > donchian_upper_aligned[i] and ema_trend_up and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Price < Donchian Lower + 1w EMA50 down + volume confirmation
            elif close_12h[i] < donchian_lower_aligned[i] and ema_trend_down and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Donchian midpoint OR 1w trend flips
            if position == 1:  # Long position
                if close_12h[i] < donchian_middle_aligned[i] or not ema_trend_up:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_12h[i] > donchian_middle_aligned[i] or not ema_trend_down:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals