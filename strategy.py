#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 Trend Filter + Volume Spike
# Long when: price breaks above 20-period Donchian high AND price > 1w EMA34 (uptrend) AND volume > 2.0x 20-period avg volume
# Short when: price breaks below 20-period Donchian low AND price < 1w EMA34 (downtrend) AND volume > 2.0x 20-period avg volume
# Uses Donchian channels for institutional support/resistance, 1w EMA for trend filter, volume spike for confirmation, discrete sizing (0.25) to minimize fee churn.
# Works in bull/bear via trend filter (avoid counter-trend) + volatility expansion (volume spike) for breakout validity.
# Timeframe: 1d (primary), HTF: 1w for EMA34 trend.

name = "1d_Donchian20_EMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_rolling
    donchian_low = low_rolling
    
    # Volume spike: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Donchian low (mean reversion)
            # 2. Price falls below 1w EMA34 (trend change)
            if (curr_close < curr_donchian_low or
                curr_close < curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Donchian high (mean reversion)
            # 2. Price rises above 1w EMA34 (trend change)
            if (curr_close > curr_donchian_high or
                curr_close > curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND above 1w EMA34 AND volume spike
            if (curr_close > curr_donchian_high and
                curr_close > curr_ema_34_1w and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below 1w EMA34 AND volume spike
            elif (curr_close < curr_donchian_low and
                  curr_close < curr_ema_34_1w and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals