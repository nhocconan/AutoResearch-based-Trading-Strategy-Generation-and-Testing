#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA trend filter and volume confirmation.
# Uses Donchian channels (20-period high/low) from daily data for breakout signals.
# Weekly EMA200 provides higher timeframe trend bias to avoid counter-trend trades.
# Volume confirmation (current volume > 1.5x 20-day average) filters low-quality breakouts.
# Designed for 1d timeframe to target 30-100 trades over 4 years.
# Works in bull/bear markets via weekly EMA trend bias and Donchian breakout logic.

name = "1d_donchian20_weekly_ema200_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA200 for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on weekly closes
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * 2 / 201) + (ema_200_1w[i-1] * 199 / 201)
    
    # Align EMA200 to daily timeframe (shifted by 1 weekly bar for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily Donchian Channel (20-period)
    # Upper band: 20-period high
    donchian_upper = np.full(n, np.nan)
    for i in range(19, n):
        donchian_upper[i] = np.max(high[i-19:i+1])
    
    # Lower band: 20-period low
    donchian_lower = np.full(n, np.nan)
    for i in range(19, n):
        donchian_lower[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after EMA200 is available
        # Skip if required data not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-day average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        if np.isnan(vol_ma):
            volume_filter = False
        else:
            volume_filter = volume[i] > vol_ma * 1.5
        
        # Trend bias: weekly EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price re-enters Donchian channel or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] < donchian_upper[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price re-enters Donchian channel or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] > donchian_lower[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of weekly trend with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian upper in uptrend
                if breakout_up and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian lower in downtrend
                elif breakout_down and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals