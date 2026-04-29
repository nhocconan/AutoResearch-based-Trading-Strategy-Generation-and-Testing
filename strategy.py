#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band width regime + Donchian(20) breakout + volume confirmation
# Bollinger Band width < 0.05 indicates low volatility/squeeze (range/accumulation).
# Donchian(20) breakout captures directional moves post-squeeze.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# 1d timeframe minimizes fee drag; discrete sizing (0.25) controls turnover.
# Works in bull/bear: squeeze breakouts occur in all regimes; volume filter avoids false breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_BBWidth_Squeeze_Donchian20_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop (1w for regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter (aligned to 1d)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Bands (20, 2) on 1d
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20  # normalized width
    
    # Donchian channels (20) on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # BB/Donchian/vol MA warmup, 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_sma_20 = sma_20[i]
        curr_bb_width = bb_width[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Regime filter: Bollinger Band width < 0.05 indicates low volatility squeeze
        squeeze = curr_bb_width < 0.05
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Breakout conditions
        breakout_long = curr_high > curr_donchian_high  # break above upper Donchian
        breakout_short = curr_low < curr_donchian_low   # break below lower Donchian
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel OR breakout fails (close below mid-band)
            if curr_low <= curr_donchian_high or curr_close < curr_sma_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel OR breakout fails (close above mid-band)
            if curr_high >= curr_donchian_low or curr_close > curr_sma_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only trade in direction of 1w EMA50 trend to avoid whipsaws
            # Long entry: squeeze + volume confirmation + upward breakout + above 1w EMA50
            if (squeeze and vol_confirm and breakout_long and 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze + volume confirmation + downward breakout + below 1w EMA50
            elif (squeeze and vol_confirm and breakout_short and 
                  curr_close < curr_ema_50_1w):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals