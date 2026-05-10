#!/usr/bin/env python3
# 1d_Keltner_Channel_Squeeze_Breakout_1wTrend
# Hypothesis: On daily timeframe, Keltner Channel squeeze (BB width < KC width) indicates low volatility.
# Breakout from squeezed KC with 1-week trend alignment (EMA50) and volume confirmation.
# Works in bull/bear by filtering breakouts with higher timeframe trend.
# Target: 15-25 trades/year, low frequency to minimize fee drag.

name = "1d_Keltner_Channel_Squeeze_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Keltner Channel (20, ATR*1.5)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean()
    kc_mid = close_series.rolling(window=20, min_periods=20).mean()
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr
    kc_width = kc_upper - kc_lower
    
    # Squeeze condition: BB width < KC width
    squeeze = bb_width < kc_width
    
    # Breakout conditions
    breakout_up = close > kc_upper
    breakout_down = close < kc_lower
    
    # Weekly trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirm = volume > volume_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 periods for BB/KC/ATR/volume
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(squeeze[i]) or np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: squeeze breakout up + weekly uptrend + volume
            if squeeze[i-1] and breakout_up[i] and uptrend_1w[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze breakout down + weekly downtrend + volume
            elif squeeze[i-1] and breakout_down[i] and downtrend_1w[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KC mid or weekly trend turns down
            if close[i] < kc_mid[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KC mid or weekly trend turns up
            if close[i] > kc_mid[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals