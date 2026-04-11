#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long: price breaks above 20-day high with 1w EMA uptrend and volume > 1.5x 20-day average
# - Short: price breaks below 20-day low with 1w EMA downtrend and volume > 1.5x 20-day average
# - Exit: price returns to opposite Donchian level (mean reversion at channel midpoint)
# - Uses 1w EMA for primary trend filter to avoid counter-trend trades
# - Works in both bull and bear markets by trading with the weekly trend
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "1d_1w_donchian_breakout_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    # Using rolling window with min_periods
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = high_rolling_max[i]
        lower_channel = low_rolling_min[i]
        
        # 1w trend: price above/below EMA
        weekly_uptrend = close_price > ema_20_1w_aligned[i]
        weekly_downtrend = close_price < ema_20_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper channel with weekly uptrend and volume
        if close_price > upper_channel and weekly_uptrend and vol_confirm:
            enter_long = True
        
        # Short breakout: price breaks below lower channel with weekly downtrend and volume
        if close_price < lower_channel and weekly_downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite channel level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to lower channel
            exit_long = close_price <= lower_channel
        elif position == -1:
            # Exit short if price rises back to upper channel
            exit_short = close_price >= upper_channel
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals