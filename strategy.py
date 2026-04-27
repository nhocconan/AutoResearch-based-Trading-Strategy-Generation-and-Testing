# 4H_WILLIAMS_FRACTAL_BREAKOUT_1D_TREND_VOLUME
# Uses Williams Fractal breakouts from daily timeframe combined with daily trend filter and volume confirmation
# Williams Fractals identify potential reversal points, breakouts from these levels with trend and volume confirmation
# capture strong momentum moves. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull markets by catching breakouts and in bear markets by following the daily trend direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low) fractals"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal: high is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: low is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = williams_fractals(high_1d, low_1d)
    
    # Williams Fractals need 2 extra bars for confirmation (point 2+2 pattern)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bear_level = bearish_fractal_aligned[i]
        bull_level = bullish_fractal_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal with uptrend and volume
            if close_val > bull_level and close_val > ema_trend and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below bearish fractal with downtrend and volume
            elif close_val < bear_level and close_val < ema_trend and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below bearish fractal (reversal signal)
            if close_val < bear_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above bullish fractal (reversal signal)
            if close_val > bull_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_WILLIAMS_FRACTAL_BREAKOUT_1D_TREND_VOLUME"
timeframe = "4h"
leverage = 1.0