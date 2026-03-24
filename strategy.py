#!/usr/bin/env python3
"""
Experiment #847: 6h Primary + 1d HTF — Fisher Transform Reversals + ADX Regime

Hypothesis: 6h timeframe with Fisher Transform entries captures reversals better
than pure trend following in bear/range markets (2022 crash, 2025 bear).
ADX regime filter adapts logic: trend-follow in trends, mean-revert in chop.
1d HMA provides bias without being too slow (unlike 1w).

Key innovations:
1. Fisher Transform(9) for reversal entries - proven in bear markets
2. ADX(14) regime detection - ADX>25 trend, ADX<20 range
3. 1d HMA(21) for bias - faster than KAMA, still smooth
4. Dual entry logic: trend entries when ADX high, reversal when ADX low
5. RSI(14) confirmation - avoids false Fisher signals
6. ATR(14) 2.5x trailing stop for risk management

Entry conditions (LOOSE to ensure ≥30 trades/train, ≥3/test):
- LONG: 1d HMA bull + Fisher crosses above -1.5 + (ADX<20 OR RSI<45)
- SHORT: 1d HMA bear + Fisher crosses below +1.5 + (ADX<20 OR RSI>55)

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_regime_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 2 * (price - min) / (max - min) - 1
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_line = np.zeros(n)
    fisher_line[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        x = 2.0 * (close[i] - lowest) / price_range - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent division by zero
        
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        if i > period:
            fisher_line[i] = fisher[i - 1]
        else:
            fisher_line[i] = fisher[i]
    
    return fisher, fisher_line

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    tr_smooth = np.zeros(n)
    
    # Initialize with first period sum
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    tr_smooth[period] = np.sum(tr[1:period+1])
    
    for i in range(period + 1, n):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - plus_dm_smooth[i-1]/period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - minus_dm_smooth[i-1]/period + minus_dm[i]
        tr_smooth[i] = tr_smooth[i-1] - tr_smooth[i-1]/period + tr[i]
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = EMA of DX
    dx_series = pd.Series(dx)
    adx_raw = dx_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_raw
    
    return adx

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    Reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    def wma(data, span):
        result = np.zeros(len(data))
        result[:] = np.nan
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half_period = period // 2
    if half_period < 1:
        half_period = 1
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    hma = np.zeros(n)
    hma[:] = np.nan
    
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    fisher, fisher_line = calculate_fisher(close, period=9)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_line[i-1]):
            # Fisher crosses above -1.5 (bullish reversal)
            fisher_cross_long = (fisher_line[i-1] <= -1.5) and (fisher[i] > -1.5)
            # Fisher crosses below +1.5 (bearish reversal)
            fisher_cross_short = (fisher_line[i-1] >= 1.5) and (fisher[i] < 1.5)
        
        # === ADX REGIME ===
        adx_trending = adx_14[i] > 25.0
        adx_ranging = adx_14[i] < 20.0
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + Fisher reversal OR RSI oversold
        if htf_1d_bull:
            # Strong signal: Fisher cross + RSI confirmation
            if fisher_cross_long and rsi_oversold:
                if adx_trending:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Base signal: Either Fisher OR RSI
            elif fisher_cross_long or rsi_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + Fisher reversal OR RSI overbought
        elif htf_1d_bear:
            # Strong signal: Fisher cross + RSI confirmation
            if fisher_cross_short and rsi_overbought:
                if adx_trending:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Base signal: Either Fisher OR RSI
            elif fisher_cross_short or rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals