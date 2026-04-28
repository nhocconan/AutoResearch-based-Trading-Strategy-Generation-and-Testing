#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike confirmation
# Elder Ray: Bull Power = High - EMA(close), Bear Power = Low - EMA(close)
# Long when Bull Power > 0 AND Bear Power < 0 (bulls in control) AND close > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when Bear Power < 0 AND Bull Power < 0 (bears in control) AND close < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when power signs flip or volume drops
# Works in bull markets by capturing sustained buying pressure, works in bear by requiring volume spikes
# which often accompany panic selling/buying climaxes that precede reversals.
# Target: 12-37 trades/year on 6h (50-150 total over 4 years).

name = "6h_ElderRay_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA(13) on 6h close for Elder Ray power calculation
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(close)
    bear_power = low - ema_13   # Bear Power = Low - EMA(close)
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 (bulls in control) AND close > 1d EMA50 AND volume confirmation
            if bp > 0 and br < 0 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND Bull Power < 0 (bears in control) AND close < 1d EMA50 AND volume confirmation
            elif br < 0 and bp < 0 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Bull Power <= 0 (bulls lose control) or volume drops
            if bp <= 0 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Bear Power >= 0 (bears lose control) or volume drops
            if br >= 0 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals