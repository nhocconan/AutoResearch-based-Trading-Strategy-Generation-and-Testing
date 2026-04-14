#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Williams %R for mean reversion in ranging markets.
# Long when daily Williams %R < -80 (oversold) and price > 12h EMA50 (trend filter).
# Short when daily Williams %R > -20 (overbought) and price < 12h EMA50.
# Exit when Williams %R crosses -50 (mean reversion complete).
# Williams %R identifies overextended moves, EMA50 filters for trend alignment to avoid counter-trend trades.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to balance signal quality and frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h EMA50
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load daily data ONCE for Williams %R
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Williams %R (14-period)
    # Highest high and lowest low over lookback period
    highest_high = np.full_like(high_daily, np.nan)
    lowest_low = np.full_like(low_daily, np.nan)
    lookback = 14
    
    for i in range(lookback - 1, len(high_daily)):
        highest_high[i] = np.max(high_daily[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low_daily[i - lookback + 1:i + 1])
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full_like(close_daily, np.nan)
    denominator = highest_high - lowest_low
    valid = (denominator != 0) & (~np.isnan(denominator))
    williams_r[valid] = ((highest_high[valid] - close_daily[valid]) / denominator[valid]) * -100
    
    # Align indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_50)
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion entries with trend filter
            # Long: oversold AND price above EMA50
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: overbought AND price below EMA50
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: mean reversion complete (Williams %R crosses -50 upward)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: mean reversion complete (Williams %R crosses -50 downward)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_WilliamsR_EMA50_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0