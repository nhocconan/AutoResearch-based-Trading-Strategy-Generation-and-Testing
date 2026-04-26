#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA20 trend filter and 1d volume spike (>2x average) captures intraday momentum with institutional participation. Works in bull/bear via trend filter. Uses discrete sizing (0.20) to minimize fee drag. Target: 60-120 trades over 4 years (15-30/year) to stay within fee limits. Close-based exits on retest of broken level.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d average volume for spike detection
    avg_volume_1d = pd.Series(df_1d['volume']).rolling(window=30, min_periods=30).mean().values  # 30d average
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate Camarilla levels from previous 1h bar
    high_1h = prices['high'].shift(1).values
    low_1h = prices['low'].shift(1).values
    close_1h = prices['close'].shift(1).values
    
    camarilla_r1 = close_1h + (high_1h - low_1h) * 1.1 / 12
    camarilla_s1 = close_1h - (high_1h - low_1h) * 1.1 / 12
    
    # ATR(14) for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of EMA(20), volume(30), ATR(14)
    start_idx = max(20, 30, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        ema_val = ema_20_4h_aligned[i]
        r1_val = camarilla_r1[i]
        s1_val = camarilla_s1[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(atr_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current 1h volume > 2x 1d average volume (scaled)
        # 1d avg volume needs to be compared to 1h volume - approximate by dividing by 26 (6.5h * 4)
        volume_confirmed = vol > (2.0 * avg_vol / 26.0)
        
        # Long: price CLOSES above R1 with 4h uptrend and volume
        long_condition = (close_val > r1_val) and (close_val > ema_val) and volume_confirmed
        # Short: price CLOSES below S1 with 4h downtrend and volume
        short_condition = (close_val < s1_val) and (close_val < ema_val) and volume_confirmed
        
        # Exit: price retests broken level (with small buffer to avoid whipsaw)
        long_exit = (position == 1 and close_val <= r1_val * 1.001)
        short_exit = (position == -1 and close_val >= s1_val * 0.999)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0