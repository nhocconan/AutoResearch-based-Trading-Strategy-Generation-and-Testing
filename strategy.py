#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and 1w volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to identify trend strength
# 1d EMA34 provides higher timeframe trend direction to avoid counter-trend trades
# 1w volume confirmation (1.5x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via bear power exhaustion + trend alignment, in bear via bull power failure
# Elder Ray works in both regimes by measuring power relative to trend, not just overbought/oversold

name = "6h_ElderRay_1dEMA34_1wVolumeS_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w volume MA for confirmation
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().shift(1).values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate Elder Ray components on 6h timeframe
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 1w volume MA
        volume_confirm = volume[i] > (vol_ma_1w_aligned[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Bear power weakening (less negative) + price > 1d EMA34 + volume confirm
            # Bear power turning up from extreme low (bullish divergence)
            if (i > start_idx and 
                bear_power[i] > bear_power[i-1] and  # Bear power increasing (less negative)
                bear_power[i-1] < np.percentile(bear_power[max(0, i-50):i], 10) and  # Was extremely weak
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bull power weakening (less positive) + price < 1d EMA34 + volume confirm
            # Bull power turning down from extreme high (bearish divergence)
            elif (i > start_idx and 
                  bull_power[i] < bull_power[i-1] and  # Bull power decreasing (less positive)
                  bull_power[i-1] > np.percentile(bull_power[max(0, i-50):i], 90) and  # Was extremely strong
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull power weakening significantly or reverse signal
            if (bull_power[i] < bull_power[i-1] and 
                bull_power[i] < np.percentile(bull_power[max(0, i-20):i], 30)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear power strengthening significantly or reverse signal
            if (bear_power[i] > bear_power[i-1] and 
                bear_power[i] > np.percentile(bear_power[max(0, i-20):i], 70)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals