#!/usr/bin/env python3
"""
1d Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Williams fractals identify key swing points on 1d. Breakouts above/below recent fractals with 1w EMA50 trend filter and volume confirmation capture momentum. Chop regime filter avoids whipsaws in ranging markets. Designed for BTC/ETH with discrete sizing (0.25) to control drawdown. Targets 30-100 trades over 4 years on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Fractals on 1d (need extra delay for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,  # Using 1w data for fractals to reduce noise
        df_1w['low'].values,
    )
    # Fractals need 2 extra 1w bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d Donchian-like levels from recent fractals (lookback 10 periods)
    lookback = 10
    # Recent bullish fractal resistance (highest bullish fractal low)
    bullish_fractal_series = pd.Series(bullish_fractal_aligned)
    resistance = bullish_fractal_series.rolling(window=lookback, min_periods=1).max().values
    # Recent bearish fractal support (lowest bearish fractal high)
    bearish_fractal_series = pd.Series(bearish_fractal_aligned)
    support = bearish_fractal_series.rolling(window=lookback, min_periods=1).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Chop regime filter: avoid trading in high chop (>61.8)
    # Using 1d ATR ratio (ATR(14)/ATR(50)) as chop proxy
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = tr.ewm(span=50, adjust=False, min_periods=50).mean().values
    chop_ratio = atr_14 / (atr_50 + 1e-10)
    chop_filter = chop_ratio < 0.618  # Low chop = trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for warmup
    start_idx = max(50, lookback, 20) + 2
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(resistance[i]) or 
            np.isnan(support[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        low_chop = chop_filter[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Fractal breakout + trend + volume + low chop
            # Long: price breaks above resistance AND bullish bias AND volume spike AND low chop
            long_entry = (curr_high > resistance[i]) and bullish_bias and vol_spike and low_chop
            # Short: price breaks below support AND bearish bias AND volume spike AND low chop
            short_entry = (curr_low < support[i]) and bearish_bias and vol_spike and low_chop
            
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
            # Exit: price falls below support (breakdown) OR loss of bullish bias OR high chop
            if (curr_low < support[i]) or (curr_close < ema_1w_aligned[i]) or (not low_chop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above resistance (breakout) OR loss of bearish bias OR high chop
            if (curr_high > resistance[i]) or (curr_close > ema_1w_aligned[i]) or (not low_chop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0