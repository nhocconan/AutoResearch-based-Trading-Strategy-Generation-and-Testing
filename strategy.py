#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Williams fractals identify significant swing highs/lows. Breakouts above recent bearish fractal (short-term resistance) or below bullish fractal (support) with volume spike and 1d EMA34 trend alignment capture momentum shifts. Works in bull/bear via 1d EMA34 trend filter (long only when price>EMA, short only when price<EMA). Designed for 75-200 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = pd.Series(high).rolling(window=1, min_periods=1).max() - pd.Series(low).rolling(window=1, min_periods=1).min()
    tr2 = abs(pd.Series(high).rolling(window=1, min_periods=1).max() - pd.Series(close).shift(1).rolling(window=1, min_periods=1).min())
    tr3 = abs(pd.Series(low).rolling(window=1, min_periods=1).min() - pd.Series(close).shift(1).rolling(window=1, min_periods=1).max())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Fractals on 1d (needs 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # ATR for dynamic stop (4h ATR)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA (34) + fractals + volume MA (20) + ATR (14)
    start_idx = max(34, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals
            long_entry = (
                curr_high > bearish_fractal_aligned[i] and  # break above bearish fractal (resistance)
                volume_spike[i] and
                curr_close > ema_34_1d_aligned[i]  # 1d uptrend filter
            )
            short_entry = (
                curr_low < bullish_fractal_aligned[i] and  # break below bullish fractal (support)
                volume_spike[i] and
                curr_close < ema_34_1d_aligned[i]  # 1d downtrend filter
            )
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: trail stop or exit on reverse signal
            # Stop loss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * atr[i]
            # Exit if price hits stop or reverse fractal break
            if curr_low <= stop_loss or curr_low < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: trail stop or exit on reverse signal
            # Stop loss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * atr[i]
            # Exit if price hits stop or reverse fractal break
            if curr_high >= stop_loss or curr_high > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0