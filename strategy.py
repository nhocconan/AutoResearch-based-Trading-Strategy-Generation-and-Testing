#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian breakouts capture momentum, EMA34 filters trend direction, volume confirms legitimacy, and chop filter avoids whipsaws in ranging markets. Designed for 4h to target 20-50 trades/year (75-200 over 4 years), minimizing fee drag. Works in both bull and bear markets by following 1d trend and avoiding counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 4h data (will be calculated in loop with min_periods)
    # Calculate 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate choppiness index (14-period) on 1d for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: use True Range and rolling max/min
    tr1 = pd.Series(df_1d['high']).rolling(14, min_periods=14).max() - pd.Series(df_1d['low']).rolling(14, min_periods=14).min()
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    high14 = pd.Series(df_1d['high']).rolling(14, min_periods=14).max()
    low14 = pd.Series(df_1d['low']).rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr14.rolling(14, min_periods=14).sum() / (np.log10(14) * (high14 - low14))) / np.log10(10)
    chop_values = chop.fillna(50).values  # fill NaN with 50 (neutral)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average (balanced for trade frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA34, volume MA
    start_idx = max(20, 34, 20)  # Donchian needs 20 bars, EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian bands for current bar (using only past data)
        lookback_start = max(0, i - 19)  # 20 bars including current
        donchian_high = np.nanmax(high[lookback_start:i+1])
        donchian_low = np.nanmin(low[lookback_start:i+1])
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor breakouts), chop > 38.2 = ranging (favor mean reversion)
        # We use chop < 61.8 to allow breakouts in trending markets
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Look for entry signals - require ALL conditions: breakout + trend + volume + regime
            # Long: price breaks above Donchian high AND bullish bias AND volume spike AND trending regime
            long_entry = (curr_high > donchian_high) and bullish_bias and vol_spike and trending_regime
            # Short: price breaks below Donchian low AND bearish bias AND volume spike AND trending regime
            short_entry = (curr_low < donchian_low) and bearish_bias and vol_spike and trending_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low (mean reversion) OR loss of bullish bias OR chop > 61.8 (ranging)
            if (curr_low < donchian_low) or (curr_close < ema_1d_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (mean reversion) OR loss of bearish bias OR chop > 61.8 (ranging)
            if (curr_high > donchian_high) or (curr_close > ema_1d_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0