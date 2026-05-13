# 4h_Keltner_Channel_Squeeze_Breakout
# Hypothesis: Keltner Channel squeeze (BBands inside KC) + volume spike triggers breakout in direction of 1d EMA200 trend. Works in bull/bear by following higher timeframe trend.
# Uses 4h timeframe with 1d trend filter, volatility squeeze, and volume confirmation for clean entries.

name = "4h_Keltner_Channel_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 4h Bollinger Bands (20, 2) ===
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # === 4h Keltner Channel (20, 1.5) ===
    atr_period = 20
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    kc_mid = bb_mid  # same as EMA/SMA middle
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr
    
    # === Squeeze condition: BB inside KC (low volatility) ===
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    # === Volume confirmation: 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_signal = 0
    
    for i in range(20, n):
        # Skip if not enough warmup for indicators
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue
            
        # Check for squeeze breakout
        if squeeze[i-1]:  # was in squeeze
            # LONG: break above BB upper with volume, in uptrend
            if high[i] > bb_upper[i] and volume_spike[i] and close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_signal = 0
            # SHORT: break below BB lower with volume, in downtrend
            elif low[i] < bb_lower[i] and volume_spike[i] and close[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_signal = 0
        
        # Exit conditions
        if position == 1:
            # Exit: price crosses below BB middle OR ATR-based trailing stop
            if low[i] < bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above BB middle OR ATR-based trailing stop
            if high[i] > bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals