#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike (>1.5x 20-bar MA), and chop regime filter (CHOP(14) < 38.2 = trending)
# Donchian breakout captures strong momentum moves. 1d EMA50 ensures alignment with higher-timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Chop filter avoids whipsaws in ranging markets.
# Discrete sizing (0.30) balances return and fee drag. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate Chopiness Index on 1d: CHOP(14) = 100 * log10(sum(ATR(1),14) / (log10(14) * (HH(14)-LL(14))))
    # Simplified: CHOP = 100 * log10( sum(tr,14) / (log10(14) * (max(high,14)-min(low,14))) )
    # We'll compute a proxy: high-low range based chop
    tr1 = np.maximum(high, np.roll(high, 1)) - np.minimum(low, np.roll(low, 1))
    tr1[0] = high[0] - low[0]  # first bar
    sum_tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(14) * (highest_high14 - lowest_low14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop = 100 * np.log10(sum_tr14 / chop_denom)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels on 4h: 20-period high/low
    highest_high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 20, 14)  # Donchian20, volume MA20, chop14
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(highest_high20[i]) or 
            np.isnan(lowest_low20[i]) or np.isnan(volume_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Chop regime filter: CHOP < 38.2 indicates trending market (good for breakouts)
        chop_trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above 20-period high, price above 1d EMA50, volume spike, trending regime
            if curr_high > highest_high20[i] and curr_close > ema_1d_50_aligned[i] and vol_spike and chop_trending:
                signals[i] = 0.30
                position = 1
            # Short: Donchian breakdown below 20-period low, price below 1d EMA50, volume spike, trending regime
            elif curr_low < lowest_low20[i] and curr_close < ema_1d_50_aligned[i] and vol_spike and chop_trending:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown below 20-period low or price below 1d EMA50
            if curr_low < lowest_low20[i] or curr_close < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout above 20-period high or price above 1d EMA50
            if curr_high > highest_high20[i] or curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals