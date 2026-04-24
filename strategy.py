#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend + RSI(2) mean reversion + 1w ADX regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for ADX regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry logic:
    * In trending regime (ADX > 25): Long when price > KAMA AND RSI(2) < 10 (pullback in uptrend).
                                 Short when price < KAMA AND RSI(2) > 90 (pullback in downtrend).
    * In ranging regime (ADX < 20): Long when RSI(2) < 15 AND price < lower Bollinger Band(20,2).
                                 Short when RSI(2) > 85 AND price > upper Bollinger Band(20,2).
- Exit: Opposite signal or RSI(2) crosses 50 (mean reversion complete).
- Signal size: 0.25 discrete to minimize fee drag.
- KAMA adapts to market noise, reducing false signals in chop.
- RSI(2) captures short-term extremes for mean reversion.
- ADX regime filter ensures we use the right strategy for market conditions.
- Works in bull markets (trend following on pullbacks) and bear markets (mean reversion in ranges, trend following on bounces).
- Estimated trades: ~60 total over 4 years (~15/year) due to regime-specific filters reducing overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average."""
    close = pd.Series(close)
    change = abs(close.diff(er_period))
    volatility = close.diff().abs().rolling(window=er_period, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = [close.iloc[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close.iloc[i] - kama[-1]))
    return np.array(kama)

def rsi(close, period=2):
    """Relative Strength Index."""
    close = pd.Series(close)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def bollinger_bands(close, period=20, std_dev=2):
    """Bollinger Bands."""
    close = pd.Series(close)
    ma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper.values, lower.values

def adx(high, low, close, period=14):
    """Average Directional Index."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).sum() / atr)
    minus_di = 100 * (abs(minus_dm.rolling(window=period, min_periods=period).sum()) / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d indicators
    kama_vals = kama(close, er_period=10, fast_sc=2, slow_sc=30)
    rsi_vals = rsi(close, period=2)
    bb_upper, bb_lower = bollinger_bands(close, period=20, std_dev=2)
    
    # Calculate 1w ADX for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    adx_1w = adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_vals[i]
        curr_adx = adx_1w_aligned[i]
        
        # Exit conditions: opposite signal or RSI(2) crosses 50 (mean reversion complete)
        if position != 0:
            # Exit long: price < KAMA OR RSI(2) > 50
            if position == 1:
                if curr_close < kama_vals[i] or curr_rsi > 50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > KAMA OR RSI(2) < 50
            elif position == -1:
                if curr_close > kama_vals[i] or curr_rsi < 50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions based on regime
        if position == 0:
            # Trending regime (ADX > 25): trend following on pullbacks
            if curr_adx > 25:
                # Long: price > KAMA AND RSI(2) < 10 (strong pullback in uptrend)
                if curr_close > kama_vals[i] and curr_rsi < 10:
                    signals[i] = 0.25
                    position = 1
                # Short: price < KAMA AND RSI(2) > 90 (strong pullback in downtrend)
                elif curr_close < kama_vals[i] and curr_rsi > 90:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime (ADX < 20): mean reversion at extremes
            elif curr_adx < 20:
                # Long: RSI(2) < 15 AND price < lower Bollinger Band (oversold)
                if curr_rsi < 15 and curr_close < bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI(2) > 85 AND price > upper Bollinger Band (overbought)
                elif curr_rsi > 85 and curr_close > bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI2_ADXRegime_v1"
timeframe = "1d"
leverage = 1.0