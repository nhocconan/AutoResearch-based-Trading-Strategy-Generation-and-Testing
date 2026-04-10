#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d trend filter and volume confirmation
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when power signals weaken or reverse
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear strength relative to trend; works in both bull and bear markets

name = "6h_1d_elder_ray_power_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d_arr = df_1d['close'].values
    ema50_1d = pd.Series(close_1d_arr).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Elder Ray components: EMA13 of close
    close_arr = prices['close'].values
    ema13 = pd.Series(close_arr).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = prices['high'].values - ema13
    bear_power = ema13 - prices['low'].values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when bull power positive, bear power negative, 1d uptrend, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and  # price above 1d EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when bear power positive, bull power negative, 1d downtrend, volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and  # price below 1d EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power weakens
            # Exit when power signals weaken or reverse
            exit_signal = False
            if position == 1:  # Long position
                if bull_power[i] <= 0 or bear_power[i] >= 0:  # bull power weak or bear power strong
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power[i] <= 0 or bull_power[i] >= 0:  # bear power weak or bull power strong
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals