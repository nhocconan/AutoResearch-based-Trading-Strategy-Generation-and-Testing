#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d close > 1d EMA50 AND 6h volume > 1.5x 20-period volume SMA
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d close < 1d EMA50 AND 6h volume > 1.5x 20-period volume SMA
# - Exit: Elder Power crosses zero or volume drops below average
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Volume spike requirement reduces false breakouts and overtrading
# - Target: 12-35 trades/year on 6h timeframe (50-140 total over 4 years) to stay within fee drag limits

name = "6h_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Calculate 6h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Elder Ray signals with momentum confirmation
        # Long: Bull Power positive AND Bear Power rising (increasing = less negative)
        # Short: Bear Power negative AND Bull Power falling (decreasing = less positive)
        if i >= 1:
            bull_power_rising = bull_power[i] > bull_power[i-1]
            bear_power_falling = bear_power[i] < bear_power[i-1]
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        long_entry = bull_power[i] > 0 and bull_power_rising and trend_bullish and vol_confirm
        short_entry = bear_power[i] < 0 and bear_power_falling and trend_bearish and vol_confirm
        
        # Exit conditions: Elder Power crosses zero or loss of volume confirmation
        exit_long = bull_power[i] <= 0 or not vol_confirm
        exit_short = bear_power[i] >= 0 or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
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