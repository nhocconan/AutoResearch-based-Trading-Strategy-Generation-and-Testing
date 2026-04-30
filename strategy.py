#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 1d volume spike for entry timing.
# Uses Supertrend(ATR=10, mult=3) on 12h to filter whipsaws and capture strong trends.
# Enters long when price breaks above recent 4h high with volume spike in uptrend.
# Enters short when price breaks below recent 4h low with volume spike in downtrend.
# Designed for low trade frequency (~20-40/year on 4h) with strong directional moves.
# Works in bull markets via trend continuation and in bear markets via shorting breakdowns.
# Volume spike (>2.0x average) ensures participation and reduces false breakouts.
# ATR-based stoploss (2.5x) and trailing exit (break of opposite 4h extreme) manages risk.

name = "4h_12hSupertrend_VolumeSpike_4hBreakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ATR(10) for Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 12h Supertrend
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + (3.0 * atr_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_12h)
    
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=float)
    direction_12h = np.ones_like(close_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        # Upper band logic
        if close_12h[i-1] <= upper_band_12h[i-1]:
            upper_band_12h[i] = min(upper_band_12h[i], upper_band_12h[i-1])
        else:
            upper_band_12h[i] = upper_band_12h[i]
        
        # Lower band logic
        if close_12h[i-1] >= lower_band_12h[i-1]:
            lower_band_12h[i] = max(lower_band_12h[i], lower_band_12h[i-1])
        else:
            lower_band_12h[i] = lower_band_12h[i]
        
        # Supertrend and direction
        if supertrend_12h[i-1] == upper_band_12h[i-1]:
            if close_12h[i] <= upper_band_12h[i]:
                supertrend_12h[i] = upper_band_12h[i]
                direction_12h[i] = -1
            else:
                supertrend_12h[i] = lower_band_12h[i]
                direction_12h[i] = 1
        else:
            if close_12h[i] >= lower_band_12h[i]:
                supertrend_12h[i] = lower_band_12h[i]
                direction_12h[i] = 1
            else:
                supertrend_12h[i] = upper_band_12h[i]
                direction_12h[i] = -1
    
    # Align 12h Supertrend direction to 4h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h.astype(float))
    
    # Calculate ATR(14) for 4h stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h recent high/low for breakout (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, lookback)  # warmup
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_trend = supertrend_dir_aligned[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h recent high with 12h uptrend
                if curr_close > curr_highest and curr_trend > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h recent low with 12h downtrend
                elif curr_close < curr_lowest and curr_trend < 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR break of 4h recent low (reversal)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lowest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR break of 4h recent high (reversal)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_highest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals