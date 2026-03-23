#!/usr/bin/env python3
"""
Experiment #255: 1h Primary + 4h/1d HTF — Simplified Trend Pullback with Volume

Hypothesis: After 200+ failed experiments, the key insight is:
1. 1h timeframe needs VERY FEW trades (30-80/year) or fees destroy profit
2. Previous 1h attempts failed due to EITHER 0 trades (#250) OR negative Sharpe (#245)
3. Solution: Use 4h/1d HMA for DIRECTION, 1h RSI for ENTRY TIMING only
4. Simplify confluence: RSI 40/60 (not extreme 15/85), volume >0.5x (not 0.8x)
5. Session filter (8-20 UTC) as soft preference, NOT hard requirement
6. Position size: 0.25 (conservative for 1h volatility)
7. ATR 2.5x trailing stoploss to limit drawdown

KEY DIFFERENCE FROM #250 (0 trades):
- RSI thresholds 40/60 instead of 35/65 (more triggers)
- Volume filter >0.5x instead of >0.8x (less restrictive)
- Session filter is soft bonus, not hard requirement
- No CHOP/CRSI complexity that caused 0-trade scenarios

TARGET: 40-80 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_4h1d_atr_simple_v1"
timeframe = "1h"
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    ratio = vol_s / (vol_avg + 1e-10)
    return ratio.fillna(1.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    hma_21_1h = calculate_hma(close, 21)
    hma_55_1h = calculate_hma(close, 55)
    atr_14_1h = calculate_atr(high, low, close, period=14)
    rsi_14_1h = calculate_rsi(close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Calculate 4h HMA for medium-term trend (aligned properly with shift(1))
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14_1h[i]) or atr_14_1h[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21_1h[i]) or np.isnan(hma_55_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h TREND (HMA crossover) ===
        hma_bullish_1h = hma_21_1h[i] > hma_55_1h[i]
        hma_bearish_1h = hma_21_1h[i] < hma_55_1h[i]
        
        # === RSI ENTRY SIGNALS (simplified - NOT extreme) ===
        # Long: RSI pullback to 40-55 zone (bullish continuation)
        rsi_pullback_long = (rsi_14_1h[i] >= 40.0) and (rsi_14_1h[i] <= 55.0)
        # Short: RSI pullback to 45-60 zone (bearish continuation)
        rsi_pullback_short = (rsi_14_1h[i] >= 45.0) and (rsi_14_1h[i] <= 60.0)
        
        # === VOLUME FILTER (soft - not mandatory) ===
        volume_ok = vol_ratio_1h[i] >= 0.5  # At least 50% of avg volume
        
        # === SESSION FILTER (soft bonus - 8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        session_ok = (hour_utc >= 8) and (hour_utc <= 20)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bullish + 4h bullish + 1h RSI pullback + volume ok
        # Session is bonus, not required
        if price_above_hma_1d and price_above_hma_4h and rsi_pullback_long and volume_ok:
            # Session bonus: increase size slightly during active hours
            if session_ok:
                desired_signal = POSITION_SIZE_FULL
            else:
                desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: 1d bearish + 4h bearish + 1h RSI pullback + volume ok
        elif price_below_hma_1d and price_below_hma_4h and rsi_pullback_short and volume_ok:
            # Session bonus: increase size slightly during active hours
            if session_ok:
                desired_signal = -POSITION_SIZE_FULL
            else:
                desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14_1h[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14_1h[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d or 4h trend turns bearish
        if in_position and position_side > 0 and (price_below_hma_1d or price_below_hma_4h):
            desired_signal = 0.0
        
        # Exit short if 1d or 4h trend turns bullish
        if in_position and position_side < 0 and (price_above_hma_1d or price_above_hma_4h):
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi_14_1h[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi_14_1h[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        # Only hold if we're in position AND no exit signal triggered
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d and 4h trend still bullish
                if price_above_hma_1d and price_above_hma_4h:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if 1d and 4h trend still bearish
                if price_below_hma_1d and price_below_hma_4h:
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