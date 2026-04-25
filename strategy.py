#!/usr/bin/env python3
"""
1d Williams Fractal Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Williams fractals identify potential reversal points. In strong weekly trends,
fractal breakouts in trend direction with volume spikes capture momentum. Works in bull/bear
by following weekly EMA34 trend filter. Targets 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Fractals on 1d (need 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Align with 2 extra delay bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 20-period volume MA for 1d volume confirmation
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, fractals, and volume MA
    start_idx = max(34 + 2, 20)  # 34+2 for EMA warmup + fractal delay, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        vol_ma_1d = vol_ma_20_1d[i]
        
        # Volume confirmation: current 1d volume > 2.5 * 20-period average
        volume_confirm = curr_volume > 2.5 * vol_ma_1d
        
        # Price breakout above recent bullish fractal (resistance) OR below bearish fractal (support)
        # Only consider fractals from completed weekly bars (already aligned)
        bullish_breakout = curr_high > bull_fractal and not np.isnan(bull_fractal)
        bearish_breakout = curr_low < bear_fractal and not np.isnan(bear_fractal)
        
        if position == 0:
            # Look for entry signals
            # Long: Bullish fractal breakout AND price > weekly EMA34 (uptrend) AND volume confirmation
            long_entry = bullish_breakout and (curr_close > ema_trend) and volume_confirm
            # Short: Bearish fractal breakout AND price < weekly EMA34 (downtrend) AND volume confirmation
            short_entry = bearish_breakout and (curr_close < ema_trend) and volume_confirm
            
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
            # Exit: price closes below weekly EMA34 OR bearish fractal breakout (contrarian signal)
            if curr_close < ema_trend or bearish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above weekly EMA34 OR bullish fractal breakout (contrarian signal)
            if curr_close > ema_trend or bullish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Fractal_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0