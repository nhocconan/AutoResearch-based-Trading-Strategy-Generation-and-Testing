# 1d_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA trend filter (adaptive moving average) on daily timeframe
# combined with RSI momentum and Choppiness index regime filter to avoid false signals.
# Works in both bull and bear markets by adapting trend detection (KAMA) and only
# trading when market is not too choppy (CHOP > 61.8 = range, < 38.2 = trend).
# Uses weekly timeframe for higher trend context to avoid counter-trend trades.
# Target: 15-25 trades/year to minimize fee drag on daily timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    # Fix: volatility should be rolling sum of absolute changes
    volatility_series = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0])))
    volatility = volatility_series.rolling(window=er_length, min_periods=1).sum().values
    change_series = pd.Series(change)
    er = np.where(volatility != 0, change_series.rolling(window=er_length, min_periods=1).sum().values / volatility, 0)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.abs(high_1d - low_1d)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((max_high - min_low) > 0, chop, 50)
    
    # Align to daily timeframe (already daily, but using for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup period for indicators
        # Skip if any data is not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2) or in strong range (CHOP > 61.8 for mean reversion)
        # We'll use CHOP < 38.2 for trend following and CHOP > 61.8 for mean reversion
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long conditions: price > KAMA (uptrend) + RSI > 50 (bullish momentum) + weekly trend up
            if is_trending and price > kama_val and rsi_val > 50 and price > ema34_1w_val:
                signals[i] = 0.25
                position = 1
            # Short conditions: price < KAMA (downtrend) + RSI < 50 (bearish momentum) + weekly trend down
            elif is_trending and price < kama_val and rsi_val < 50 and price < ema34_1w_val:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging markets: buy at support, sell at resistance
            elif is_ranging:
                # Buy when RSI oversold (< 30) and price near KAMA (support)
                if rsi_val < 30 and price <= kama_val * 1.005:  # Within 0.5% of KAMA
                    signals[i] = 0.25
                    position = 1
                # Sell when RSI overbought (> 70) and price near KAMA (resistance)
                elif rsi_val > 70 and price >= kama_val * 0.995:  # Within 0.5% of KAMA
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when trend changes or momentum fades
                if (price < kama_val or  # Trend broken
                    rsi_val < 40 or      # Momentum lost
                    (is_ranging and rsi_val > 60)):  # In range, RSI overbought
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when trend changes or momentum fades
                if (price > kama_val or  # Trend broken
                    rsi_val > 60 or      # Momentum lost
                    (is_ranging and rsi_val < 40)):  # In range, RSI oversold
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "1d"
leverage = 1.0