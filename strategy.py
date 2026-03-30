Looking at the experiment history, I see a clear pattern: too many strategies fail with 0 trades (too strict) or negative Sharpe (wrong logic). The winning strategies from DB use **price channel structure (Donchian/Camarilla) + volume confirmation + regime filter**.

For 12h, I need entries that trigger **every 2-4 weeks** to hit the 50-150 target over 4 years. Let me design a simple, proven approach with tight entry conditions.

**Hypothesis**: Williams %R at extremes (±80/±20) captures reversals, but ONLY when combined with ATR expansion (volatility breakout) and 1d trend alignment. This catches the "volatility squeeze" reversals that work in both bull and bear markets.

**Why it should work in bull AND bear**: Williams %R extremes + ATR expansion = capitulation/reversal zones. Works whether market is trending up or down. 1d EMA keeps us on the right side of the larger trend.
#!/usr/bin/env python3
"""
Experiment #021: 12h Williams %R + ATR Expansion + 1d EMA Trend

HYPOTHESIS: Williams %R at extremes (±80/±20) captures reversals, but ONLY when
combined with ATR expansion (volatility breakout = institutional move exhaustion).
1d EMA keeps us aligned with the larger trend.

WHY 12h: 3x fewer trades than 4h = less fee drag. ATR expansion on 12h captures
multi-day volatility regimes.

TARGET: 50-100 total trades over 4 years = 12-25/year. HARD MAX: 150.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_atr_exp_ema50_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_williams_r(high, low, close, period=14):
    """Williams %R"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        if highest != lowest:
            willr[i] = -100 * (highest - close[i]) / (highest - lowest)
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # ATR ratio for volatility expansion detection
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=20).mean().values
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    last_trade_bar = -100  # Anti-churn: minimum 5 bars between trades
    
    warmup = 150  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(willr_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK: Need volatility expansion ===
        # Skip if ATR ratio < 1.0 (not enough volatility for signal)
        if atr_ratio[i] < 1.0:
            if in_position:
                # Still manage existing position
                pass
            else:
                signals[i] = 0.0
                continue
        
        # === TREND DIRECTION (1d EMA50) ===
        bull_trend = close[i] > ema_1d_aligned[i]
        bear_trend = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === WILLIAMS %R SIGNALS ===
        # Oversold = potential reversal up (bullish)
        willr_oversold = willr_14[i] <= -80
        # Overbought = potential reversal down (bearish)
        willr_overbought = willr_14[i] >= -20
        
        # === ATR EXPANSION CONFIRMATION ===
        atr_expansion = atr_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        bars_since_trade = i - last_trade_bar
        
        if not in_position and bars_since_trade >= 5:
            # === LONG: Williams %R oversold + ATR expansion + bull trend + volume ===
            if willr_oversold and atr_expansion and bull_trend and vol_spike:
                desired_signal = SIZE
                last_trade_bar = i
            
            # === SHORT: Williams %R overbought + ATR expansion + bear trend + volume ===
            if willr_overbought and atr_expansion and bear_trend and vol_spike:
                desired_signal = -SIZE
                last_trade_bar = i
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (4x ATR from entry) ===
        if in_position:
            if position_side > 0:
                profit_target = entry_price + 4.0 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = 0.0
            else:
                profit_target = entry_price - 4.0 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals