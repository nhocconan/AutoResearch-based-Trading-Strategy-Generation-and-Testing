#!/usr/bin/env python3
"""
Experiment #263: 1d Primary + 1w HTF — Donchian Breakout with Weekly Trend Filter

Hypothesis: After 200+ failed experiments with complex regime-switching, return to proven
simple breakout strategy that worked on SOL (Sharpe +0.782) but adapt for BTC/ETH:
- 1w HMA(21) for ultra-long-term trend bias (smoother than daily CHOP)
- 1d Donchian(20) breakout for entry signals (proven on daily timeframe)
- RSI(14) filter at 40/60 (not extreme 30/70) to avoid chasing breakouts
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25-0.30 discrete levels

KEY DIFFERENCE FROM FAILED #253, #257:
- No CHOP regime-switching (caused -2.297 Sharpe on 1d)
- Weekly HMA is smoother than daily CHOP for trend detection
- Donchian breakout triggers more reliably than CRSI extremes
- RSI 40/60 filter (not 30/70) allows more signals while avoiding extremes

TARGET: 20-50 trades/year on 1d, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1w HMA for ultra-long-term trend (aligned properly with shift(1))
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price breaks above 20-day high
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # break prev bar's upper
        # Short: price breaks below 20-day low
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # break prev bar's lower
        
        # === RSI FILTER (avoid chasing extremes) ===
        # Long: RSI not overbought (< 65)
        rsi_ok_long = rsi_14[i] < 65.0
        # Short: RSI not oversold (> 35)
        rsi_ok_short = rsi_14[i] > 35.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1w bullish + Donchian breakout + RSI filter
        if price_above_hma_1w and donchian_breakout_long and rsi_ok_long:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY: 1w bearish + Donchian breakout + RSI filter
        elif price_below_hma_1w and donchian_breakout_short and rsi_ok_short:
            desired_signal = -POSITION_SIZE_FULL
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1w trend turns bearish
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        # Exit short if 1w trend turns bullish
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (> 70)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (< 30)
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        # Only hold if we're in position AND no exit signal triggered
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1w trend still bullish
                if price_above_hma_1w:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if 1w trend still bearish
                if price_below_hma_1w:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals