#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) mean reversion + choppiness regime filter
# Long when: KAMA upward (trending up) AND RSI < 30 (oversold) AND choppy market (CHOP > 61.8)
# Short when: KAMA downward (trending down) AND RSI > 70 (overbought) AND choppy market (CHOP > 61.8)
# Exit when: RSI crosses back to neutral (40-60 range) OR choppy regime ends (CHOP < 38.2)
# Uses discrete position sizing (0.25) to minimize fee drag.
# Target: 30-70 trades total over 4 years (7-17/year) on 1d.
# KAMA adapts to market noise, reducing false signals in chop.
# RSI extremes work well in mean-reverting choppy markets (chop filter ensures we only mean revert in chop).
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) regimes.

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - ER=10, Fast=2, Slow=30
    # ER = Efficiency Ratio = |net change| / sum of absolute changes
    # Smooth = [ER * (fastest - slowest) + slowest]^2
    # KAMA[i] = KAMA[i-1] + smooth * (price[i] - KAMA[i-1])
    def calculate_kama(price, er_period=10, fast=2, slow=30):
        n = len(price)
        kama = np.full(n, np.nan)
        if n < er_period + 1:
            return kama
        
        # Direction over er_period
        direction = np.abs(np.diff(price, n=er_period))
        
        # Volatility: sum of absolute changes over er_period
        volatility = np.nansum(np.abs(np.diff(price, n=1))[np.newaxis, :], axis=0)
        volatility = np.concatenate([np.full(er_period, np.nan), volatility])
        
        # Avoid division by zero
        er = np.where(volatility != 0, direction / volatility, 0)
        
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Initialize KAMA
        kama[er_period] = price[er_period]
        
        # Calculate KAMA
        for i in range(er_period + 1, n):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        
        return kama
    
    # Calculate Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(TR over n) / (max(high,n) - min(low,n))) / log10(n)
    def calculate_chop(high, low, close, n_period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First TR
        
        # Sum of TR over n_period
        tr_sum = np.convolve(tr, np.ones(n_period), 'full')[:len(tr)]
        tr_sum = np.where(np.arange(len(tr_sum)) < n_period-1, np.nan, tr_sum)
        
        # Max high and min low over n_period
        max_high = np.convolve(high, np.ones(n_period), 'full')[:len(high)]
        max_high = np.where(np.arange(len(max_high)) < n_period-1, np.nan, max_high)
        min_low = np.convolve(low, np.ones(n_period), 'full')[:len(low)]
        min_low = np.where(np.arange(len(min_low)) < n_period-1, np.nan, min_low)
        
        # Avoid division by zero and invalid values
        denominator = max_high - min_low
        chop = np.full(len(high), np.nan)
        valid = (denominator > 0) & (~np.isnan(tr_sum)) & (~np.isnan(denominator))
        chop[valid] = 100 * np.log10(tr_sum[valid] / denominator[valid]) / np.log10(n_period)
        
        return chop
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period), 'full')[:len(gain)]
        avg_loss = np.convolve(loss, np.ones(period), 'full')[:len(loss)]
        
        # First values
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Smooth subsequent values
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # Set first period values to NaN
        rsi[:period] = np.nan
        
        return rsi
    
    # Calculate indicators
    kama = calculate_kama(close)
    chop = calculate_chop(high, low, close)
    rsi = calculate_rsi(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Warmup for KAMA and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]) or 
            i < 1):  # Need previous KAMA for trend
            signals[i] = 0.0
            continue
        
        # KAMA trend: upward if current KAMA > previous KAMA
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Chop regime: choppy market (good for mean reversion)
        choppy = chop[i] > 61.8
        trending = chop[i] < 38.2
        
        vol_close = close[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: RSI returns to neutral OR chop regime ends (trending)
            if rsi_neutral[i] or trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral OR chop regime ends (trending)
            if rsi_neutral[i] or trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long: KAMA up + RSI oversold + choppy market
            if kama_up and rsi_oversold and choppy:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + choppy market
            elif kama_down and rsi_overbought and choppy:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals