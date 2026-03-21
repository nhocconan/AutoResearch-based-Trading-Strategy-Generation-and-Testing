#!/usr/bin/env python3
"""
Hypothesis: 12h DEMA trend + 1d Donchian regime + RSI pullback entries
- 12h primary timeframe captures multi-day swings without noise
- 1d Donchian breakout defines major trend regime (bull/bear/range)
- DEMA (Double EMA) faster than HMA, better for 12h entries
- RSI pullback (40-60 zone) for entries with trend, not extremes
- BB Width percentile detects squeeze (mean reversion) vs expansion (trend)
- ATR trailing stop at 2.5*ATR for risk management
- Asymmetric sizing: bigger longs in bull regime, smaller shorts in bear
Timeframe: 12h (primary), 1d (HTF regime filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_donchian_regime_12h_v1"
timeframe = "12h"
leverage = 1.0

def calculate_dema(close, span=21):
    """Double Exponential Moving Average - faster response than EMA"""
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=span, min_periods=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, min_periods=span, adjust=False).mean()
    dema = 2 * ema1 - ema2
    return dema.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - high/low of last N periods"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Fill initial values
    for i in range(period-1):
        upper[i] = np.max(high[:i+1])
        lower[i] = np.min(low[:i+1])
    
    return upper, lower

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Bollinger Band Width as % of price"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return width, sma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian for major trend regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donch_upper_1d, donch_lower_1d = calculate_donchian(high_1d, low_1d, period=20)
    donch_mid_1d = (donch_upper_1d + donch_lower_1d) / 2
    
    # Align 1d Donchian to 12h
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    
    # 12h indicators
    dema_fast = calculate_dema(close, span=10)
    dema_slow = calculate_dema(close, span=25)
    atr = calculate_atr(high, low, close, period=14)
    bb_width, bb_mid = calculate_bb_width(close, period=20, std_mult=2.0)
    rsi = calculate_rsi(close, period=14)
    
    # BB Width percentile (rolling 100 bars)
    bb_width_s = pd.Series(bb_width)
    bb_percentile = bb_width_s.rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x) if len(x) > 0 else 0.5, raw=False
    ).values
    bb_percentile = np.nan_to_num(bb_percentile, 0.5)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.32   # 32% for longs in bull regime
    SIZE_SHORT = 0.22  # 22% for shorts in bear regime (asymmetric)
    SIZE_NEUTRAL = 0.15  # Small position in range
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF regime from 1d Donchian
        price_vs_donch = (close[i] - donch_mid_aligned[i]) / (donch_upper_aligned[i] - donch_lower_aligned[i] + 1e-8)
        
        # Regime: >0.6 = bull, <0.4 = bear, 0.4-0.6 = range
        if price_vs_donch > 0.6:
            regime = 1  # Bull
        elif price_vs_donch < 0.4:
            regime = -1  # Bear
        else:
            regime = 0  # Range
        
        # 12h trend from DEMA crossover
        dema_trend = 1.0 if dema_fast[i] > dema_slow[i] else -1.0
        
        # BB regime: <0.3 = squeeze (mean reversion), >0.7 = expansion (trend)
        bb_regime = bb_percentile[i]
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first - MUST exit on stop
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop or close[i] < entry_price - atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop or close[i] > entry_price + atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry logic - regime adaptive
        signal_set = False
        
        if regime == 1:  # Bull regime - prefer longs
            if dema_trend > 0 and 35 < rsi[i] < 60:
                # Trend up + RSI pullback (not overbought)
                if bb_regime > 0.3:  # Some volatility
                    signals[i] = SIZE_LONG
                    if prev_signal == 0:
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = close[i]
                    signal_set = True
            elif rsi[i] > 72:
                # Overbought - exit longs
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                signal_set = True
        
        elif regime == -1:  # Bear regime - prefer shorts or flat
            if dema_trend < 0 and 40 < rsi[i] < 65:
                # Trend down + RSI pullback (not oversold)
                if bb_regime > 0.3:  # Some volatility
                    signals[i] = -SIZE_SHORT
                    if prev_signal == 0:
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = close[i]
                    signal_set = True
            elif rsi[i] < 28:
                # Oversold - exit shorts
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                signal_set = True
        
        else:  # Range regime - mean reversion
            if bb_regime < 0.35:  # Squeeze - expect breakout
                if dema_trend > 0 and rsi[i] < 50:
                    signals[i] = SIZE_NEUTRAL
                    if prev_signal == 0:
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = close[i]
                    signal_set = True
                elif dema_trend < 0 and rsi[i] > 50:
                    signals[i] = -SIZE_NEUTRAL
                    if prev_signal == 0:
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = close[i]
                    signal_set = True
        
        if not signal_set:
            signals[i] = prev_signal
        
        prev_signal = signals[i]
    
    return signals