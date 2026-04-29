#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d EMA34 trend filter + volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13; strong when power expands with trend.
# Works in bull markets via bull power expansion above EMA13; in bear markets via bear power expansion below EMA13.
# Volume spike confirms institutional participation. Target: 12-25 trades/year (50-100 total).

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v2"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strong institutional interest)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # warmup for 1d EMA, EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish regime: price above 1d EMA34
            if curr_close > curr_ema34_1d:
                # Long when bull power expands (increasing) + volume spike
                if i > start_idx and curr_bull_power > bull_power[i-1] and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
            # Bearish regime: price below 1d EMA34
            elif curr_close < curr_ema34_1d:
                # Short when bear power expands (more negative) + volume spike
                if i > start_idx and curr_bear_power < bear_power[i-1] and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: bull power contracts OR price breaks below 1d EMA34
            if curr_bull_power < bull_power[i-1] or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: bear power contracts (less negative) OR price breaks above 1d EMA34
            if curr_bear_power > bear_power[i-1] or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals