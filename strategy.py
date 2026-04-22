#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and 1d volume confirmation
# Donchian breakout captures breakouts in trending markets.
# Weekly EMA(34) filter ensures we only trade in the direction of the higher timeframe trend.
# 1d volume spike (>1.8x 20-day avg) confirms breakout strength and avoids false signals.
# Works in bull markets by capturing upward breakouts and in bear markets by capturing downward breakdowns
# with volume confirmation preventing whipsaws. Targets 12-30 trades/year on 6f timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Load daily data for volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Donchian Channel (20) on 6h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band + weekly uptrend + volume spike
            if (close[i] > highest_high_20[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.8 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band + weekly downtrend + volume spike
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long on return to lower band or weekly trend turns down
                if (close[i] < lowest_low_20[i] or 
                    close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short on return to upper band or weekly trend turns up
                if (close[i] > highest_high_20[i] or 
                    close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA34_1dVolSpike"
timeframe = "6h"
leverage = 1.0