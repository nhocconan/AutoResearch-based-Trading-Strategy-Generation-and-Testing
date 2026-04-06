#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + daily volume confirmation + 1w trend filter.
# Uses daily volume > 1.5x 20-day average for confirmation and weekly EMA(50) for trend direction.
# Long when price breaks above Donchian high (20) and weekly trend is up.
# Short when price breaks below Donchian low (20) and weekly trend is down.
# Includes ATR-based stoploss (2.5x ATR) to manage risk.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian20_daily_vol_weekly_trend_v1"
timeframe = "12h"
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
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    volume_filter = volume > (vol_ma_20d_aligned * 1.5)
    
    # Weekly trend filter: EMA(50) on weekly chart
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_50w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr
            
            if close[i] < donchian_low[i] or close[i] < stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr
            
            if close[i] > donchian_high[i] or close[i] > stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter[i]:
                # Long: price breaks above Donchian high and weekly trend up
                if close[i] > donchian_high[i] and close[i] > ema_50w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low and weekly trend down
                elif close[i] < donchian_low[i] and close[i] < ema_50w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals