#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime_v1
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d trend filter, volume spike, and chop regime filter captures strong directional moves while avoiding choppy markets. Uses discrete sizing (0.25) and strict volume confirmation (2.0x) to limit trades to 19-50/year. Works by only taking breakouts aligned with 1d trend in trending regimes (CHOP < 38.2), reducing whipsaw in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous day's OHLC for Camarilla calculation (use 1d data)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter: CHOP < 38.2 = trending (use 14-period)
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop_regime = chop < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA34 (34), volume MA (20), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        trend_val = ema34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        regime = chop_regime[i]
        
        # Skip if any data not ready
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(trend_val) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Camarilla breakout in direction of trend + volume + regime
        long_condition = (close_val > r1_val) and is_uptrend and vol_conf and regime
        short_condition = (close_val < s1_val) and is_downtrend and vol_conf and regime
        
        # Exit conditions: opposite Camarilla level touch or trend reversal or regime change
        long_exit = (position == 1 and (close_val < s1_val or not is_uptrend or not regime))
        short_exit = (position == -1 and (close_val > r1_val or not is_downtrend or not regime))
        
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

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0