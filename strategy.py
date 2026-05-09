#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w EMA200 trend filter and 4h Choppiness Index regime.
# Long when price > 1w EMA200 and CHOP > 61.8 (range), short when price < 1w EMA200 and CHOP > 61.8.
# Mean-reversion in range, trend-following in trending markets (CHOP < 38.2) with trend filter.
# Designed for low trade frequency (<50/year) to avoid fee drag in 4h timeframe.
# Works in bull/bear markets by adapting to regime and using multi-timeframe trend.
name = "4h_Choppiness_Regime_1wEMA200"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Choppiness Index (14-period) on 4h data
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's smoothing for ATR
        atr[period] = np.sum(tr[1:period+1]) / period
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Sum of true range over period
        sum_tr = np.zeros(len(high))
        for i in range(period, len(high)):
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
        # Choppiness Index
        chop = np.zeros(len(high))
        for i in range(period, len(high)):
            if sum_tr[i] > 0 and atr[i] > 0:
                chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * period)) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14)  # Wait for EMA200 and CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_200_4h[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_value = chop[i]
        price_above_trend = close[i] > ema_200_4h[i]
        price_below_trend = close[i] < ema_200_4h[i]
        
        # Regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
        in_range = chop_value > 61.8
        in_trend = chop_value < 38.2
        
        if position == 0:
            # In range: mean reversion
            if in_range:
                if price_above_trend:
                    # Sell high in range (expect pullback)
                    signals[i] = -0.25
                    position = -1
                elif price_below_trend:
                    # Buy low in range (expect bounce)
                    signals[i] = 0.25
                    position = 1
            # In trend: follow trend with filter
            elif in_trend:
                if price_above_trend:
                    # Uptrend: buy
                    signals[i] = 0.25
                    position = 1
                elif price_below_trend:
                    # Downtrend: sell
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: opposite signal or regime change to extreme trend
            if price_below_trend or (in_range and chop_value < 50) or (not in_range and not in_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite signal or regime change to extreme trend
            if price_above_trend or (in_range and chop_value > 50) or (not in_range and not in_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals