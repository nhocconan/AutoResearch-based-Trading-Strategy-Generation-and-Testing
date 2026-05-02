#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# We use 1d EMA34 for higher-timeframe trend alignment (avoid counter-trend trades)
# Volume spike (>2.0 x 20-period EMA) confirms institutional participation
# Discrete position sizing (0.25) controls fee drag
# Works in bull markets by taking long signals with bullish trend, works in bear by only taking short signals with bearish trend
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (EMA34) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 1d volume confirmation: volume > 2.0 x 20-period EMA
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ema_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 34, 20, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d: price > EMA34 = bullish trend
        bullish_trend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (bulls in control) + volume spike + bullish 1d trend
            if bull_power[i] > 0 and volume_spike_1d_aligned[i] and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + volume spike + bearish 1d trend
            elif bear_power[i] < 0 and volume_spike_1d_aligned[i] and not bullish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bulls lose control (Bull Power <= 0) OR trend turns bearish
            if bull_power[i] <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bears lose control (Bear Power >= 0) OR trend turns bullish
            if bear_power[i] >= 0 or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals