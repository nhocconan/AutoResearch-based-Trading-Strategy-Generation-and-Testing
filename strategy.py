#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long: 12h close > Donchian upper(20) + price > 1w EMA50 (uptrend) + 12h volume > 1.5x 20-period MA
# - Short: 12h low < Donchian lower(20) + price < 1w EMA50 (downtrend) + 12h volume > 1.5x 20-period MA
# - Exit: Close back inside Donchian channel (mean reversion) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Donchian breakouts capture volatility expansion, EMA50 filters for higher-timeframe trend,
#   volume confirms institutional participation. Works in bull/bear: breakouts with trend in bull,
#   mean reversion exits in bear ranges.

name = "12h_1w_donchian_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channel (20-period) for 12h
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume moving average (20-period)
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_20_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma_20_12h)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for Donchian20)
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h OHLCV
        open_price = open_12h[i]
        high_price = high_12h[i]
        low_price = low_12h[i]
        close_price = close_12h[i]
        volume_price = volume_12h[i]
        
        # Get aligned 1w data for current 12h bar (completed 1w bar)
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_price > 1.5 * volume_ma_current
        
        # Donchian breakout conditions
        bullish_breakout = close_price > donchian_upper[i]
        bearish_breakout = low_price < donchian_lower[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish Donchian breakout + price > 1w EMA50 + volume spike
            if (bullish_breakout and close_price > ema_50_current and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish Donchian breakout + price < 1w EMA50 + volume spike
            elif (bearish_breakout and close_price < ema_50_current and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price closes back inside the Donchian channel (mean reversion)
            if position == 1 and close_price < donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price > donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals