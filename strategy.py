#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout + 1d EMA34 trend + volume confirmation
# Williams fractals identify swing highs/lows; breakout above recent bearish fractal or below bullish fractal
# with 1d EMA34 trend filter and volume spike captures momentum in both bull and bear markets.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag.

name = "6h_WilliamsFractal_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Williams Fractals (5-bar: 2 left, 2 right)
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Align fractals with 2-bar delay for confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 34, 14, 20)  # warmup for EMA34, ATR, volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_bearish_fractal = bearish_fractal_aligned[i]
        curr_bullish_fractal = bullish_fractal_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_price = entry_price - 2.0 * atr_at_entry
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Price crosses below 1d EMA34 (trend change)
            # 3. Price re-enters below bullish fractal (breakout failed)
            if (curr_low <= stop_price or
                curr_close < curr_ema_34_1d or
                curr_close < curr_bullish_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_price = entry_price + 2.0 * atr_at_entry
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Price crosses above 1d EMA34 (trend change)
            # 3. Price re-enters above bearish fractal (breakout failed)
            if (curr_high >= stop_price or
                curr_close > curr_ema_34_1d or
                curr_close > curr_bearish_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above nearest bearish fractal + above 1d EMA34 + volume confirm
            if (not np.isnan(curr_bearish_fractal) and
                curr_close > curr_bearish_fractal and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: price breaks below nearest bullish fractal + below 1d EMA34 + volume confirm
            elif (not np.isnan(curr_bullish_fractal) and
                  curr_close < curr_bullish_fractal and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals