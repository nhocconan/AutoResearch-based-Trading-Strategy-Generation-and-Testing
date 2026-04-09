#!/usr/bin/env python3
# daily_cci_volume_reversal_v1
# Hypothesis: Daily CCI reversals with volume confirmation and weekly trend filter work in both bull and bear markets.
# Uses CCI(20) > 100 for overbought (short) and < -100 for oversold (long) with volume > 1.5x 20-day average.
# Weekly EMA50 trend filter ensures alignment with higher timeframe trend.
# Target: 15-25 trades/year (60-100 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_cci_volume_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate CCI (Commodity Channel Index)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - sma_tp) / (0.015 * mad)
    cci = cci.values
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Weekly trend filter: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(cci[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI > -100 (oversold condition exhausted) or weekly trend turns bearish
            if cci[i] > -100 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI < 100 (overbought condition exhausted) or weekly trend turns bullish
            if cci[i] < 100 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI < -100 (oversold) with volume confirmation and weekly uptrend
            if cci[i] < -100 and volume[i] > vol_threshold[i] and close[i] > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI > 100 (overbought) with volume confirmation and weekly downtrend
            elif cci[i] > 100 and volume[i] > vol_threshold[i] and close[i] < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals