#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
Williams Fractals identify potential reversal points. We trade breakouts of recent fractal highs/lows
only when aligned with the 1d trend (EMA50) and confirmed by volume spikes.
This combines mean-reversion fractal signals with trend-following filters to work in both bull and bear markets.
Target: 50-150 trades over 4 years (12-37/year) to avoid excessive fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams Fractals on 6h data (look for recent fractals)
    # Need minimum 5 points for fractal detection
    if len(high) < 5 or len(low) < 5:
        return np.zeros(n)
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    
    # For breakout trading, we need the most recent fractal levels
    # We'll use the last bearish fractal as resistance and bullish as support
    # Initialize arrays to hold the most recent fractal levels
    recent_bearish = np.full(n, np.nan)  # resistance levels
    recent_bullish = np.full(n, np.nan)  # support levels
    
    # Forward fill the most recent fractal values
    last_bearish = np.nan
    last_bullish = np.nan
    for i in range(n):
        if not np.isnan(bearish_fractal[i]):
            last_bearish = bearish_fractal[i]
        if not np.isnan(bullish_fractal[i]):
            last_bullish = bullish_fractal[i]
        recent_bearish[i] = last_bearish
        recent_bullish[i] = last_bullish
    
    # Volume spike detection on 6h (current bar volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    # Start from index 20 to ensure indicators are ready
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(recent_bearish[i]) or 
            np.isnan(recent_bullish[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend: price above/below EMA50
        uptrend_1d = close_1d[-1] > ema_50_1d[-1] if len(close_1d) > 0 else False  # Use last known 1d value
        # More robust: use aligned 1d EMA and approximate current 1d close
        # We'll use the 6h close vs 1d EMA as trend proxy
        trending_up = close[i] > ema_50_1d_aligned[i]
        trending_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Breakout conditions
        # Long: price breaks above recent bearish fractal (resistance) in uptrend with volume
        breakout_long = (close[i] > recent_bearish[i] and 
                         trending_up and 
                         vol_confirm and
                         not np.isnan(recent_bearish[i]))
        
        # Short: price breaks below recent bullish fractal (support) in downtrend with volume
        breakout_short = (close[i] < recent_bullish[i] and 
                          trending_down and 
                          vol_confirm and
                          not np.isnan(recent_bullish[i]))
        
        # Exit conditions: return to the opposite fractal level or trend change
        exit_long = (position == 1 and 
                     (close[i] < recent_bullish[i] or not trending_up))
        exit_short = (position == -1 and 
                      (close[i] > recent_bearish[i] or not trending_down))
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_fractal_breakout"
timeframe = "6h"
leverage = 1.0