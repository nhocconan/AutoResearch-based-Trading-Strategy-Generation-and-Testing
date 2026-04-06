#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout with Weekly EMA Filter and Volume Confirmation
# Uses daily Donchian(20) breakouts for trend direction, filtered by weekly EMA(50) to avoid counter-trend trades.
# Volume confirmation (current volume > 1.5x 50-period average) ensures institutional participation.
# ATR-based stoploss (2.5x ATR) manages risk in volatile markets.
# Works in bull/bear markets: breakouts capture trends, EMA filter avoids whipsaws.
# Target: 50-100 trades over 4 years (12-25/year).

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA for trend filter (50-period)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on weekly data
    ema_50 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50[49] = np.mean(close_1w[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50[i] = (close_1w[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align weekly EMA to daily timeframe (shifted by 1 weekly bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 50-period average
    vol_ma = np.full(n, np.nan)
    for i in range(49, n):
        vol_ma[i] = np.mean(volume[i-49:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr
            
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] < donchian_low[i] or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr
            
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] > donchian_high[i] or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and EMA filter
            if volume_filter:
                # Long: breakout above Donchian high with price above weekly EMA
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low with price below weekly EMA
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals