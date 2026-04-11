#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# - Long: price breaks above 20-day Donchian high with weekly EMA uptrend and volume spike
# - Short: price breaks below 20-day Donchian low with weekly EMA downtrend and volume spike
# - Exit: price returns to opposite Donchian level (mean reversion at channel midpoint)
# - Uses 1w EMA for trend filter to avoid counter-trend trades in choppy markets
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by combining breakout momentum with trend alignment

name = "1d_1w_donchian_breakout_trend_v1"
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
    if len(df_1w) < 2:
        return signals
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values  # Prior 20 bars
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values   # Prior 20 bars
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Donchian levels
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        middle_channel = (upper_channel + lower_channel) / 2
        
        # Weekly trend filter
        weekly_uptrend = ema_21_1w_aligned[i] > close_price
        weekly_downtrend = ema_21_1w_aligned[i] < close_price
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper Donchian with weekly uptrend and volume
        if close_price > upper_channel and weekly_uptrend and vol_confirm:
            enter_long = True
        
        # Short breakout: price breaks below lower Donchian with weekly downtrend and volume
        if close_price < lower_channel and weekly_downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite Donchian level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to middle/lower channel
            exit_long = close_price <= middle_channel
        elif position == -1:
            # Exit short if price rises back to middle/upper channel
            exit_short = close_price >= middle_channel
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals