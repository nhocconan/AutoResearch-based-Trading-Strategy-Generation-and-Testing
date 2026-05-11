#/usr/bin/env python3
name = "6h_FairValueGap_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load 1d data ONCE for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 2. Load 1d data ONCE for Fair Value Gap detection (bullish/bearish imbalances)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish FVG: low[i] > high[i-2] (gap up)
    bullish_fvg = np.zeros(len(high_1d), dtype=bool)
    bearish_fvg = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(high_1d)):
        if low_1d[i] > high_1d[i-2]:
            bullish_fvg[i] = True
        if high_1d[i] < low_1d[i-2]:
            bearish_fvg[i] = True
    
    # 3. Align FVG signals to 6h timeframe
    bullish_fvg_aligned = align_htf_to_ltf(prices, df_1d, bullish_fvg.astype(float))
    bearish_fvg_aligned = align_htf_to_ltf(prices, df_1d, bearish_fvg.astype(float))
    
    # 4. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 5. Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bullish_fvg_aligned[i]) or 
            np.isnan(bearish_fvg_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        bullish_fvg_signal = bullish_fvg_aligned[i] > 0.5
        bearish_fvg_signal = bearish_fvg_aligned[i] > 0.5
        
        if position == 0:
            # Long: Bullish FVG + above 1d EMA34 + volume spike
            if bullish_fvg_signal and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Bearish FVG + below 1d EMA34 + volume spike
            elif bearish_fvg_signal and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - reverse FVG or trend reversal
            if position == 1:
                # Exit: Bearish FVG appears OR price crosses below EMA
                if bearish_fvg_signal or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Bullish FVG appears OR price crosses above EMA
                if bullish_fvg_signal or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals