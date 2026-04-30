#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume confirmation
# Targets 60-120 total trades over 4 years (15-30/year). Discrete sizing 0.25.
# Uses 1d EMA50 to filter trend direction and avoid counter-trend whipsaws in bear markets.
# Volume confirmation requires current ATR-scaled volume > 1.5x 20-period median to ensure institutional participation.
# Designed to work in both bull (breakout continuation) and bear (mean reversion at extremes) regimes via trend filter.

name = "6h_Donchian20_1dEMA50_ATRVol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for volatility normalization and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR-scaled volume: volume / ATR (normalizes for volatility)
    atr_safe = np.where(atr_14 == 0, 1e-10, atr_14)  # avoid division by zero
    vol_atr = volume / atr_safe
    # Median of vol_atr over 20 periods (more stable than mean for volume)
    vol_atr_median = pd.Series(vol_atr).rolling(window=20, min_periods=20).median().values
    volume_confirmation = vol_atr > (1.5 * vol_atr_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 50, 14, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_atr_median[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr_14[i]
        curr_volume_confirm = volume_confirmation[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation to avoid low-participation breakouts
            if curr_volume_confirm:
                # Bullish: Close breaks above 20-period high AND close above 1d EMA50 (uptrend)
                if curr_close > curr_high_20 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below 20-period low AND close below 1d EMA50 (downtrend)
                elif curr_close < curr_low_20 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry (wider to avoid noise)
            stop_loss = entry_price - 2.5 * curr_atr
            # Exit: Stoploss hit OR close drops below 20-period low OR loses 1d uptrend
            if curr_low <= stop_loss or curr_close < curr_low_20 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * curr_atr
            # Exit: Stoploss hit OR close rises above 20-period high OR loses 1d downtrend
            if curr_high >= stop_loss or curr_close > curr_high_20 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals