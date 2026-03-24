# Strategy: mtf_4h_kama_trend_1d_bias_chop_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.677 | -18.4% | -35.7% | 769 | FAIL |
| ETHUSDT | -0.727 | -29.2% | -43.6% | 767 | FAIL |
| SOLUSDT | 0.030 | +9.4% | -46.5% | 731 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.552 | +17.6% | -17.4% | 244 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #339: 4h Primary + 1d HTF — Adaptive Trend with Regime Filter

Hypothesis: Previous 4h failures (#329, #331, #334) had issues with:
1. Too many regime filters killing trade count
2. Symmetric long/short logic (BTC/ETH bear markets need asymmetric approach)
3. CRSI mean reversion doesn't work well in sustained trends

This strategy uses:
1. 1d KAMA(21) as MACRO BIAS (hard filter: only long if 1d bullish, only short if 1d bearish)
2. 4h KAMA(10/30) crossover for trend entries (more adaptive than HMA)
3. 4h Choppiness Index for regime (CHOP>58=avoid trend entries, wait for mean revert)
4. 4h Volume spike confirmation for breakouts (volume > 1.5x 20-bar avg)
5. ATR-based trailing stop (2.5x ATR) + take profit at 3R

KEY INSIGHT: Asymmetric entries based on 1d trend. In bull macro, only take longs.
In bear macro, only take shorts. This reduces whipsaw in 2022 crash and 2025 bear.

TARGET: 25-45 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_trend_1d_bias_chop_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(er_period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close_s.iloc[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # KAMA for trend (adaptive)
    kama_fast_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Volume moving average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 1d KAMA for macro bias (HARD FILTER)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h
    
    # Position tracking for stoploss/takeprofit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_fast_4h[i]) or np.isnan(kama_slow_4h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d KAMA - HARD FILTER) ===
        # Only take LONGS if price above 1d KAMA (bullish macro)
        # Only take SHORTS if price below 1d KAMA (bearish macro)
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 58.0  # High choppiness = avoid trend entries
        is_trending = chop[i] < 42.0  # Low choppiness = trend friendly
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # === TREND SIGNAL (4h KAMA crossover) ===
        kama_bullish = kama_fast_4h[i] > kama_slow_4h[i]
        kama_bearish = kama_fast_4h[i] < kama_slow_4h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending or (not is_choppy):
            # TREND REGIME: KAMA crossover entries
            
            # LONG: 1d bullish + 4h KAMA bullish crossover + volume confirmation
            if price_above_kama_1d and kama_bullish:
                # Entry on pullback in uptrend (RSI 40-60) or breakout (RSI > 60 + volume)
                if (40 <= rsi_14[i] <= 65) or (rsi_14[i] > 60 and volume_confirmed):
                    desired_signal = BASE_SIZE
            
            # SHORT: 1d bearish + 4h KAMA bearish crossover + volume confirmation
            elif price_below_kama_1d and kama_bearish:
                # Entry on pullback in downtrend (RSI 35-60) or breakdown (RSI < 40 + volume)
                if (35 <= rsi_14[i] <= 60) or (rsi_14[i] < 40 and volume_confirmed):
                    desired_signal = -BASE_SIZE
        
        else:
            # CHOPPY REGIME: Mean reversion at extremes
            # Only take trades aligned with 1d bias
            
            if price_above_kama_1d and rsi_14[i] < 30:
                # Long oversold in bullish macro
                desired_signal = BASE_SIZE * 0.7  # Smaller size in chop
            
            elif price_below_kama_1d and rsi_14[i] > 70:
                # Short overbought in bearish macro
                desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (3R target) ===
        take_profit_triggered = False
        
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 3.0 * entry_atr:
                take_profit_triggered = True
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 3.0 * entry_atr:
                take_profit_triggered = True
        
        if take_profit_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (overbought/oversold reversal) ===
        if in_position and position_side > 0 and rsi_14[i] > 75:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered and not take_profit_triggered:
            # Check if trend still valid
            if position_side > 0 and kama_bullish and price_above_kama_1d:
                desired_signal = BASE_SIZE
            elif position_side < 0 and kama_bearish and price_below_kama_1d:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 08:48
