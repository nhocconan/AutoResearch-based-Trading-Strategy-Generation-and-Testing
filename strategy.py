#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# Long when price breaks above upper BB, weekly EMA(8) rising, and volume > 2x average
# Short when price breaks below lower BB, weekly EMA(8) falling, and volume > 2x average
# Exit when price returns to middle BB or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Bollinger Band squeeze (low volatility) as entry condition
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_bb_squeeze_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=8, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    ma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    middle_bb = ma_20
    
    # Bollinger Band width for squeeze detection
    bb_width = (upper_bb - lower_bb) / middle_bb
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < 0.8 * bb_width_ma  # Bollinger Band squeeze
    
    # 1d volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or breaks below lower BB
            elif close[i] <= middle_bb[i] or close[i] < lower_bb[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or breaks above upper BB
            elif close[i] >= middle_bb[i] or close[i] > upper_bb[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and Bollinger Band squeeze
            # Long: price breaks above upper BB, weekly EMA rising, volume spike, BB squeeze
            if (close[i] > upper_bb[i] and
                ema_1w_aligned[i] > ema_1w_aligned[i-1] and  # EMA rising
                volume[i] > 2.0 * volume_ma_1d_aligned[i] and
                squeeze_condition[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB, weekly EMA falling, volume spike, BB squeeze
            elif (close[i] < lower_bb[i] and
                  ema_1w_aligned[i] < ema_1w_aligned[i-1] and  # EMA falling
                  volume[i] > 2.0 * volume_ma_1d_aligned[i] and
                  squeeze_condition[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals