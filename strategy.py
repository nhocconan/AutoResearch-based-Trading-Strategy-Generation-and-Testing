#!/usr/bin/env python3
"""
Experiment #003: 1h Multi-Timeframe Regime-Adaptive Strategy with 4h HMA Bias
Hypothesis: 1h timeframe captures intermediate trends while avoiding 15m noise. 
Uses 4h HMA for primary trend bias, 1h Bollinger Band Width for regime detection.
REGIME-ADAPTIVE: Trending (BBW > 60th percentile) = trend follow on pullbacks.
                 Ranging (BBW < 40th percentile) = mean revert at BB bounds.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25-0.35)
with 2.5*ATR stoploss. Must work on BTC/ETH (not just SOL) through 2022 crash and 2025 bear.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_hma_bb_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    pct = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, sma, width, pct

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_percentile_rank(values, window=100):
    """Calculate percentile rank of current value over rolling window."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window, n):
        window_vals = values[i-window:i]
        current = values[i]
        rank = np.sum(window_vals < current) / window
        pr[i] = rank
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma, bb_width, bb_pct = calculate_bollinger(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate BB Width percentile for regime detection
    bb_width_pct_rank = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    SIZE_QUARTER = 0.08
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(bb_width_pct_rank[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection via BB Width percentile
        # > 0.6 = trending regime, < 0.4 = ranging regime
        regime_trending = bb_width_pct_rank[i] > 0.55
        regime_ranging = bb_width_pct_rank[i] < 0.45
        
        # Bollinger positions
        bb_near_lower = bb_pct[i] < 0.15
        bb_near_upper = bb_pct[i] > 0.85
        bb_below_lower = close[i] < bb_lower[i]
        bb_above_upper = close[i] > bb_upper[i]
        
        # RSI zones - more relaxed for more trades
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # ADX trend strength - lower threshold for more trades
        trend_moderate = adx[i] > 18
        trend_strong = adx[i] > 25
        
        # EMA for additional trend confirmation
        ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending regime + 4h bullish + RSI pullback
        if regime_trending and hma_4h_bullish and rsi_neutral and trend_moderate:
            new_signal = SIZE_ENTRY
        
        # Path 2: Ranging regime + BB lower + RSI oversold (mean reversion)
        elif regime_ranging and bb_near_lower and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h bullish + BB below lower (overshoot long)
        elif hma_4h_bullish and bb_below_lower and rsi_extreme_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 4: EMA bullish + RSI oversold bounce
        elif ema_bullish and rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h bullish + ADX building (trend starting)
        elif hma_4h_bullish and adx[i] > 15 and adx[i] > adx[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 6: BB near lower + RSI turning up (any regime)
        elif bb_near_lower and rsi[i] > rsi[i-1] if i > 0 else False and rsi[i] < 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending regime + 4h bearish + RSI pullback
        if regime_trending and hma_4h_bearish and rsi_neutral and trend_moderate:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Ranging regime + BB upper + RSI overbought (mean reversion)
        elif regime_ranging and bb_near_upper and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h bearish + BB above upper (overshoot short)
        elif hma_4h_bearish and bb_above_upper and rsi_extreme_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 4: EMA bearish + RSI overbought drop
        elif ema_bearish and rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h bearish + ADX building (trend starting)
        elif hma_4h_bearish and adx[i] > 15 and adx[i] > adx[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 6: BB near upper + RSI turning down (any regime)
        elif bb_near_upper and rsi[i] < rsi[i-1] if i > 0 else False and rsi[i] > 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
                # Take profit at 3R - reduce to quarter
                elif profit >= 3.0 * risk and position_reduced:
                    new_signal = SIZE_QUARTER
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
                # Take profit at 3R - reduce to quarter
                elif profit >= 3.0 * risk and position_reduced:
                    new_signal = -SIZE_QUARTER
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals