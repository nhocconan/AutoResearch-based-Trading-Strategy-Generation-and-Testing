#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Fractals identify key swing highs/lows from 1d chart. 
Breakouts above bearish fractals (H4) or below bullish fractals (L4) with 
1d EMA50 trend alignment and volume spike capture strong directional moves. 
Works in bull markets via breakout continuation and in bear markets via 
mean-reversion from extreme levels when 1d trend opposes price action.
Designed for 12h timeframe targeting 12-37 trades/year. Uses discrete 
position sizing (0.25) to minimize fee churn. Uses proper MTF loading 
with get_htf_data called once before loop and additional_delay_bars=2 
for fractal confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Fractals need 2 extra 1d bars for confirmation (center bar + 2 future bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss on 12h data
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_1d, ATR, and volume MA to propagate
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_1d = ema_50_1d_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above Bearish Fractal (H4) AND uptrend (price > 1d EMA50) AND volume spike
            long_condition = (curr_close > bearish_fractal_val) and (curr_close > ema50_1d) and volume_spike
            # Short: price breaks below Bullish Fractal (L4) AND downtrend (price < 1d EMA50) AND volume spike
            short_condition = (curr_close < bullish_fractal_val) and (curr_close < ema50_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below Bullish Fractal (L4) (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < bullish_fractal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above Bearish Fractal (H4) (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > bearish_fractal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0