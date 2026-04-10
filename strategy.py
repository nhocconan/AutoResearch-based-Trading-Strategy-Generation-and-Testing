#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close)
# - Long when Bull Power > 0 AND 1w close > 1w EMA34 AND volume > 1.5x 20-bar average
# - Short when Bear Power < 0 AND 1w close < 1w EMA34 AND volume > 1.5x 20-bar average
# - Exit when power reverses sign OR volume drops below 0.7x average
# - Uses 1w trend filter to avoid counter-trend trades in all market regimes
# - Moderate volume threshold balances signal quality and trade frequency (target: 12-30 trades/year)
# - Works in bull (trend continuation) and bear (mean reversion from extremes) markets

name = "6h_1w_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data properly
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Align them to 6h timeframe
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(34) for trend filter
    ema34_1w = pd.Series(c_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute Elder Ray components on 6h timeframe
    close_prices = prices['close'].values
    high_prices = prices['high'].values
    low_prices = prices['low'].values
    
    # Calculate EMA13 of close for Elder Ray
    ema13_close = pd.Series(close_prices).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close)
    bull_power = high_prices - ema13_close
    bear_power = low_prices - ema13_close
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND 1w uptrend AND volume spike
            if (bull_power[i] > 0 and 
                c_1w_aligned[i] > ema34_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND 1w downtrend AND volume spike
            elif (bear_power[i] < 0 and 
                  c_1w_aligned[i] < ema34_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Power reverses sign (trend exhaustion)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (bull_power[i] <= 0 or vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (bear_power[i] >= 0 or vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals