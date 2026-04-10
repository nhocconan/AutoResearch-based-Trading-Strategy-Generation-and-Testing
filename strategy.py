#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# - Weekly pivot levels: breakout above weekly R2 = long, below weekly S2 = short
# - Volume confirmation: current 6h volume > 1.5x 20-period EMA (avoid low-volume fakeouts)
# - Exit: opposite weekly pivot level (S2/R2 reversal) or max hold of 3 bars (18h)
# - Position sizing: 0.25 discrete level
# - Session filter: 08-20 UTC to avoid Asian session noise
# - Targets ~20-40 trades/year on 6h timeframe. Uses weekly pivot for structure,
#   volume confirmation reduces whipsaws, Donchian breakout captures momentum.

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_in_trade = 0  # track time in trade for max hold
    
    # Calculate weekly pivot levels (using previous week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # R2 = P + (H - L), S2 = P - (H - L)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    pivot_point = (prev_high + prev_low + prev_close) / 3
    weekly_range = prev_high - prev_low
    weekly_r2 = pivot_point + weekly_range
    weekly_s2 = pivot_point - weekly_range
    
    # Calculate 6h volume EMA for confirmation
    volume_ema_20_6h = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(weekly_r2[i]) or np.isnan(weekly_s2[i]) or 
            np.isnan(volume_ema_20_6h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            bars_in_trade = 0  # reset if forced flat
            continue
        
        # HTF volume confirmation: 6h volume > 1.5x its 20-period EMA
        vol_confirm_6h = volume[i] > 1.5 * volume_ema_20_6h[i]
        
        # Entry conditions
        long_entry = (close[i] > weekly_r2[i] and vol_confirm_6h)
        short_entry = (close[i] < weekly_s2[i] and vol_confirm_6h)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                bars_in_trade = 0
            elif short_entry:
                position = -1
                signals[i] = -0.25
                bars_in_trade = 0
            else:
                signals[i] = 0.0
                bars_in_trade = 0
        else:  # Have position - look for exit
            bars_in_trade += 1
            # Exit conditions: opposite weekly pivot level or max hold (3 bars = 18h)
            if position == 1:  # Long position
                if (close[i] < weekly_s2[i] or  # opposite pivot level
                    bars_in_trade >= 3):        # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (close[i] > weekly_r2[i] or  # opposite pivot level
                    bars_in_trade >= 3):        # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = -0.25
    
    return signals