#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 12h Camarilla R1/S1 breakouts filtered by 1d EMA34 trend and volume spike (>2.0x 20-bar MA). Uses Bollinger Band Width regime filter (BBW < 50th percentile = low volatility range) to avoid whipsaws in chop. R1/S1 are tighter breakout levels than R4/S4, increasing trade frequency but with strong volume and regime confirmation to maintain quality. Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years). Works in bull/bear markets by following 1d trend while using Camarilla structure for precise entries. Volume spike and BBW regime filter reduce false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels (using 1d for structure)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (tighter breakout levels)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 12)  # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 12)  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Bollinger Band Width on 1d for regime filter (low volatility = range)
    bb_period = 20
    bb_std = 2.0
    sma_1d = pd.Series(close_1d_vals).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d_vals).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + (bb_std * std_1d)
    lower_bb = sma_1d - (bb_std * std_1d)
    bb_width = (upper_bb - lower_bb) / sma_1d * 100  # Percentage
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # BBW percentile rank (250-day lookback for regime)
    bbw_percentile = pd.Series(bb_width).rolling(window=250, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bbw_percentile_aligned = align_htf_to_ltf(prices, df_1d, bbw_percentile)
    low_volatility_regime = bbw_percentile_aligned < 0.5  # BBW < 50th percentile = low vol range
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 34 for 1d EMA, 250 for BBW percentile, 1 for camarilla)
    start_idx = max(20, 34, 250)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(low_volatility_regime[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        low_vol_regime = low_volatility_regime[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla R1/S1 in trend direction with volume spike AND low volatility regime
        long_entry = (close_val > camarilla_r1_val) and bullish_1d and vol_spike and low_vol_regime
        short_entry = (close_val < camarilla_s1_val) and bearish_1d and vol_spike and low_vol_regime
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point or not bullish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point or not bearish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0