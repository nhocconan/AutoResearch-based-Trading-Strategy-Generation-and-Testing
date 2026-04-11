#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_v1
Strategy: 1d KAMA trend with RSI momentum and Choppiness index regime filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: In low-chop regimes (trending markets), go long when KAMA turns up and RSI > 50; short when KAMA turns down and RSI < 50. In high-chop regimes (ranging markets), stay flat. Uses weekly trend filter to avoid counter-trend trades. Designed for both bull and bear markets by adapting to market regime - trend following in trending markets, avoiding chop. Low-frequency design targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators ===
    # KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle first element
        volatility = np.concatenate([[np.sum(np.abs(np.diff(close[:period])))], volatility])
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.full_like(close, np.nan, dtype=float)
        kama_vals[period] = close[period]
        for i in range(period+1, len(close)):
            if not np.isnan(sc[i]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    # RSI
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.concatenate([[np.mean(gain[:period])], np.zeros(len(close)-period)])
        avg_loss = np.concatenate([[np.mean(loss[:period])], np.zeros(len(close)-period)])
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Choppiness Index
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max(tr1[0], tr2[0], tr3[0])], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.convolve(tr, np.ones(period)/period, mode='same')
        atr[:period-1] = np.nan
        atr[-1] = np.nan
        # Sum of ATR over period
        atr_sum = np.convolve(tr, np.ones(period)/period, mode='same') * period
        atr_sum[:period-1] = np.nan
        atr_sum[-1] = np.nan
        # Price range over period
        max_high = np.concatenate([[np.max(high[:period])], np.zeros(len(high)-period)])
        min_low = np.concatenate([[np.min(low[:period])], np.zeros(len(low)-period)])
        for i in range(period, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        price_range = max_high - min_low
        # Choppiness
        cpi = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        return cpi
    
    # Calculate indicators
    kama_vals = kama(close, period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    cpi_vals = choppiness_index(high, low, close, period=14)
    
    # === 1w Trend Filter ===
    close_1w = df_1w['close'].values
    # Weekly EMA for trend direction
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False).values
    # Shift to avoid look-ahead (use previous week's close)
    ema_1w_shifted = np.roll(ema_1w, 1)
    ema_1w_shifted[0] = np.nan
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_shifted)
    
    # Align 1d indicators (though they're already on 1d timeframe, we ensure proper alignment)
    kama_aligned = kama_vals
    rsi_aligned = rsi_vals
    cpi_aligned = cpi_vals
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(cpi_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: Choppiness > 61.8 = ranging (avoid), < 38.2 = trending (trade)
        ranging_market = cpi_aligned[i] > 61.8
        trending_market = cpi_aligned[i] < 38.2
        
        # Weekly trend filter: only trade in direction of weekly trend
        price_above_weekly_ema = close[i] > ema_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # KAMA direction: slope of KAMA
        kama_rising = kama_aligned[i] > kama_aligned[i-1] if i > 0 else False
        kama_falling = kama_aligned[i] < kama_aligned[i-1] if i > 0 else False
        
        # RSI momentum
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        # Long conditions: trending market + KAMA rising + RSI > 50 + price above weekly EMA
        long_signal = trending_market and kama_rising and rsi_above_50 and price_above_weekly_ema
        
        # Short conditions: trending market + KAMA falling + RSI < 50 + price below weekly EMA
        short_signal = trending_market and kama_falling and rsi_below_50 and price_below_weekly_ema
        
        # Exit conditions: reverse signal or chop increases
        exit_long = position == 1 and (not kama_rising or rsi_aligned[i] < 50 or ranging_market)
        exit_short = position == -1 and (not kama_falling or rsi_aligned[i] > 50 or ranging_market)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: In low-chop regimes (trending markets), go long when KAMA turns up and RSI > 50; short when KAMA turns down and RSI < 50. In high-chop regimes (ranging markets), stay flat. Uses weekly trend filter to avoid counter-trend trades. Designed for both bull and bear markets by adapting to market regime - trend following in trending markets, avoiding chop. Low-frequency design targets 15-25 trades/year to minimize fee drag.