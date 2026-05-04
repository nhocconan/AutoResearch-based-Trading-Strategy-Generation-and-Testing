#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI + 1d Weekly Trend Filter
# Uses 6h VW-RSI(14) for mean reversion signals in ranging markets, filtered by 1d weekly EMA(34) trend direction.
# VW-RSI reduces false signals during low-volume periods and improves signal quality.
# Weekly trend filter ensures we only take mean-reversion trades in the direction of the higher timeframe trend,
# reducing whipsaw during strong trends. Designed for 12-30 trades/year (~50-120 total over 4 years).
# Works in bull/bear markets by adapting to weekly trend: long VW-RSI oversold in uptrend, short VW-RSI overbought in downtrend.

name = "6h_VolWeightedRSI_1dWeeklyEMA_TrendFilter"
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
    
    # Get 1d data for weekly trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 (weekly trend proxy - ~1.5 months)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly trend to 6h timeframe (wait for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 6h Volume-Weighted RSI
    # Typical Price = (H+L+C)/3
    typical_price = (high + low + close) / 3.0
    # Volume-weighted typical price change
    vwtp = typical_price * volume
    # Price change
    price_change = np.diff(typical_price, prepend=typical_price[0])
    # Volume-weighted price change
    vw_price_change = price_change * volume
    
    # Separate gains and losses
    gains = np.where(vw_price_change > 0, vw_price_change, 0)
    losses = np.where(vw_price_change < 0, -vw_price_change, 0)
    
    # Calculate smoothed averages using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First average is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current_value) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gains = wilders_smoothing(gains, 14)
    avg_losses = wilders_smoothing(losses, 14)
    
    # Calculate RS and RSI
    rs = np.divide(avg_gains, avg_losses, out=np.full_like(avg_gains, np.nan), where=avg_losses!=0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vw_rsi[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend direction: price above/below EMA34
        weekly_uptrend = close_1d[-1] > ema34_1d_aligned[i] if len(close_1d) > 0 else False  # Use current 1d close for trend
        # More robust: compare current 1d close to its EMA (need to get current 1d close properly)
        # Simpler approach: use the aligned EMA value and compare to 1d close series
        
        # Get current 1d close for trend comparison (last completed 1d bar)
        # Since we're in 6h loop, we need to find the corresponding 1d bar
        # Use the fact that align_htf_to_ltf gives us the value from the last completed 1d bar
        # We'll use a simpler trend filter: 6h price vs 6h EMA50 for additional confirmation
        
        # Calculate 6h EMA50 for additional trend filter
        if i >= 50:
            ema50_6h = pd.Series(close[:i+1]).ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
        else:
            ema50_6h = np.nan
        
        # Skip if EMA50 not ready
        if np.isnan(ema50_6h):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # 6h trend filter: price above/below EMA50
        uptrend_6h = close[i] > ema50_6h
        downtrend_6h = close[i] < ema50_6h
        
        if position == 0:
            # Long conditions: VW-RSI oversold (<30) AND weekly uptrend AND 6h uptrend
            if (vw_rsi[i] < 30 and 
                close[i] > ema34_1d_aligned[i] and  # Weekly trend filter: price above weekly EMA
                uptrend_6h):
                signals[i] = 0.25
                position = 1
            # Short conditions: VW-RSI overbought (>70) AND weekly downtrend AND 6h downtrend
            elif (vw_rsi[i] > 70 and 
                  close[i] < ema34_1d_aligned[i] and  # Weekly trend filter: price below weekly EMA
                  downtrend_6h):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VW-RSI returns to neutral (50) or weekly trend turns down
            if (vw_rsi[i] >= 50 or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VW-RSI returns to neutral (50) or weekly trend turns up
            if (vw_rsi[i] <= 50 or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals