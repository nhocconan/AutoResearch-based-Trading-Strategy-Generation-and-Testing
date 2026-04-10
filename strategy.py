#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
# - Williams %R(14) measures overbought/oversold levels (-20 to -80 range)
# - Long when %R crosses above -80 from below AND 1w close > 1w EMA50 (bullish trend) AND volume > 2x 20-period volume SMA
# - Short when %R crosses below -20 from above AND 1w close < 1w EMA50 (bearish trend) AND volume > 2x 20-period volume SMA
# - Exit: %R crosses opposite threshold (-20 for long exit, -80 for short exit) or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Williams %R works well in ranging markets (2025-2026 bear/range) and catches reversals in trends

name = "6h_1w_williamsr_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 6h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 2x 20-period volume SMA (spike detection)
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Williams %R signals (with hysteresis to avoid whipsaw)
        # Long: %R crosses above -80 from below (oversold bounce)
        wr_long_signal = (williams_r[i-1] <= -80) and (williams_r[i] > -80)
        # Short: %R crosses below -20 from above (overbought rejection)
        wr_short_signal = (williams_r[i-1] >= -20) and (williams_r[i] < -20)
        
        # Exit conditions: %R crosses opposite threshold
        exit_long = williams_r[i] >= -20  # %R crosses above -20 (overbought)
        exit_short = williams_r[i] <= -80  # %R crosses below -80 (oversold)
        
        if position == 0:  # Flat - look for entry
            if wr_long_signal and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif wr_short_signal and trend_bearish and vol_confirm:
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