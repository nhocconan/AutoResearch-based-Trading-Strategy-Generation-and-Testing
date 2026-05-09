#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w MA trend filter + 1d RSI mean reversion
# In choppy markets (CHOP > 61.8), RSI extremes revert to mean; in trending markets (CHOP < 38.2), follow trend
# Weekly MA filter ensures we only take trades in direction of higher timeframe trend
# Designed to work in both bull (trend following) and bear (mean reversion in chop) markets
# Target: 15-25 trades/year to minimize fee drag

name = "1d_ChopRegime_RSI_MeanRev_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        # Wilder's smoothing for ATR
        atr[period] = np.sum(tr[1:period+1]) / period
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate highest high and lowest low over period
        highest_high = np.zeros(len(close))
        lowest_low = np.zeros(len(close))
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.full(len(close), 50.0)
        for i in range(period, len(close)):
            if highest_high[i] != lowest_low[i]:
                log_sum = np.sum(np.log(atr[i-period+1:i+1] / (highest_high[i] - lowest_low[i])))
                chop[i] = 100 * np.log10(np.exp(log_sum) / np.log(period)) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close)
    
    # Calculate RSI (14-period)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[0] = 50  # neutral for first value
        return rsi
    
    rsi = calculate_rsi(close)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        
        # Regime detection
        is_choppy = chop_val > 61.8  # Range-bound market
        is_trending = chop_val < 38.2  # Trending market
        
        if position == 0:
            # Long conditions
            if is_choppy and rsi_val < 30:  # Oversold in chop = mean reversion long
                signals[i] = 0.25
                position = 1
            elif is_trending and close[i] > ema50_1w_val:  # Uptrend = follow trend
                signals[i] = 0.25
                position = 1
            # Short conditions
            elif is_choppy and rsi_val > 70:  # Overbought in chop = mean reversion short
                signals[i] = -0.25
                position = -1
            elif is_trending and close[i] < ema50_1w_val:  # Downtrend = follow trend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 in chop, or price below weekly EMA in trend
            if (is_choppy and rsi_val > 50) or (is_trending and close[i] < ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 in chop, or price above weekly EMA in trend
            if (is_choppy and rsi_val < 50) or (is_trending and close[i] > ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals