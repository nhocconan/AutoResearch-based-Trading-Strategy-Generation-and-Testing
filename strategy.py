#!/usr/bin/env python3
"""
Experiment #2207: 6h Williams %R Extreme + 1d Trend Filter + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold extremes on 6h timeframe.
- Primary: 6h Williams %R(14) with extreme thresholds (<10 for long, >90 for short) 
- HTF: 1d EMA(50) trend filter (only trade counter-trend to daily extreme reversals)
- Volume: Require volume > 1.5x 20-bar average to confirm exhaustion
- Exit: Opposite Williams %R level (%R > 50 for longs, %R < 50 for shorts) or 3*ATR stop
- Target: 75-150 total trades over 4 years (19-37/year) - balanced for 6h timeframe
- Designed to work in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2207_6h_williamsr_extreme_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Trend: 1 if close > EMA (uptrend), -1 otherwise (downtrend)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20), ATR(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    # Avoid division by zero
    denominator = highest_high - lowest_low
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / denominator[mask]) * -100
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long
                # Exit if Williams %R returns above 50 (momentum fading)
                if williams_r[i] > 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price moves 3*ATR against position
                elif price < entry_price - 3.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if Williams %R returns below 50 (momentum fading)
                if williams_r[i] < 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price moves 3*ATR against position
                elif price > entry_price + 3.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Williams %R extremely oversold (<10) AND 1d trend down (bear market bounce)
            if williams_r[i] < 10 and trend_1d_aligned[i] < 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Williams %R extremely overbought (>90) AND 1d trend up (bull market top)
            elif williams_r[i] > 90 and trend_1d_aligned[i] > 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals