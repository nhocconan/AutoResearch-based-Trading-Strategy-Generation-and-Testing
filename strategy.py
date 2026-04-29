#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d EMA34 trend filter + volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13; trend filter ensures trades with higher timeframe momentum;
# volume spike confirms institutional participation. Works in bull/bear by taking longs in uptrends when bull power > 0,
# shorts in downtrends when bear power < 0. Target: 12-25 trades/year (50-100 total).

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 34-period EMA on 1d close for trend filter
    df_1d_close = df_1d['close']
    ema34_1d = pd.Series(df_1d_close.values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (to avoid noise)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 34)  # warmup for 1d EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_spike = volume_spike[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Uptrend (price > 1d EMA34): look for longs when bull power positive + volume spike
            if curr_close > curr_ema34_1d:
                if curr_bull_power > 0 and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
            # Downtrend (price < 1d EMA34): look for shorts when bear power negative + volume spike
            elif curr_close < curr_ema34_1d:
                if curr_bear_power < 0 and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when bull power fades or trend breaks
            if curr_bull_power <= 0 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when bear power fades or trend breaks
            if curr_bear_power >= 0 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals