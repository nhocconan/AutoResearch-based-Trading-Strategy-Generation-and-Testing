#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Price_Action_Reversal_v1"
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
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d High/Low for price action levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's high and low
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # === 4h Bollinger Bands for volatility and mean reversion ===
    bb_period = 20
    bb_std = 2.0
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # === 4h RSI for momentum confirmation ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Align 1d levels to 4h timeframe ===
    prev_high_1d_4h = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_4h = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = bb_period  # warmup for Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(prev_high_1d_4h[i]) or np.isnan(prev_low_1d_4h[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for reversal at previous day's levels with RSI confirmation
            # Long: price near previous day's low, RSI oversold
            long_cond = (close[i] <= prev_low_1d_4h[i] * 1.002 and  # Within 0.2% of low
                        rsi[i] < 30)
            
            # Short: price near previous day's high, RSI overbought
            short_cond = (close[i] >= prev_high_1d_4h[i] * 0.998 and  # Within 0.2% of high
                         rsi[i] > 70)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches Bollinger middle or RSI overbought
            if close[i] >= bb_middle[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches Bollinger middle or RSI oversold
            if close[i] <= bb_middle[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Price action reversal strategy that looks for reversals at previous day's
# high/low levels with RSI confirmation. Works in both bull and bear markets by
# capturing mean reversion at key daily levels. Uses 4h Bollinger Bands and RSI for
# exit timing. Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Effective on BTC/ETH as institutions
# often defend previous day's high/low levels.