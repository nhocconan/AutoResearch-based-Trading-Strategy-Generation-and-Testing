#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d Williams Fractal regime filter + 1d volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF 1d: Williams Fractal identifies swing highs/lows for regime (bullish if above recent bull fractal, bearish if below recent bear fractal).
- Volume: Current 6h volume > 1.8 * 20-period 1d volume MA to avoid low-volume breakouts.
- Entry: Long when price breaks above Donchian(20) high AND bullish regime AND volume spike.
         Short when price breaks below Donchian(20) low AND bearish regime AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Williams Fractal requires 2-bar confirmation delay after the center bar.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for Williams Fractal, EMA34 trend, and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate EMA(34) on 1d close for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        
        # Determine regime: bullish if price above recent bull fractal, bearish if below recent bear fractal
        # If no fractal found, use EMA trend as fallback
        if not np.isnan(bull_fractal) and curr_close > bull_fractal:
            regime = 1  # bullish
        elif not np.isnan(bear_fractal) and curr_close < bear_fractal:
            regime = -1  # bearish
        else:
            regime = 1 if ema_34_val > 0 and curr_close > ema_34_val else (-1 if ema_34_val > 0 and curr_close < ema_34_val else 0)
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper Donchian AND bullish regime
                if curr_high > upper_donchian and regime == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian AND bearish regime
                elif curr_low < lower_donchian and regime == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR loss of volume confirmation
            if curr_low < lower_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR loss of volume confirmation
            if curr_high > upper_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dWilliamsFractalRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0