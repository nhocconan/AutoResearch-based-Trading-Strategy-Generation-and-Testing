#!/usr/bin/env python3
"""
Experiment #005: 12h RSI(2) Extreme Reversal + 1d Trend Filter

HYPOTHESIS: RSI(2) at extreme levels (<20 or >80) marks local reversal points
where short-term mean reversion is likely. The 1d SMA(50) provides a trend bias
filter that prevents fading major trends.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: RSI(2)<20 + price>1d_SMA50 → strong long reversal setups
- Bear markets: RSI(2)>80 + price<1d_SMA50 → strong short reversal setups
- Range markets: RSI(2) extremes work as pure mean-reversion signals
- The 1d filter adapts to regime (bull=long bias, bear=short bias)

TARGET: 75-150 total trades over 4 years (tight entry conditions).
Timeframe: 12h primary, 1d for trend filter.
Signal: 0.25-0.30 (discrete).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi2_extreme_reversal_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend direction
    sma_1d_raw = calculate_sma(df_1d['close'].values, 50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # 1d EMA(21) for smoother trend
    ema_1d_raw = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # 12h indicators
    rsi_2 = calculate_rsi(close, 2)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup - need enough for 1d SMA alignment
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(rsi_2[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND CHECK (1d) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_above_1d_ema = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else price_above_1d_sma
        
        # Strong trend = price above both SMA and EMA
        bullish_trend = price_above_1d_sma and price_above_1d_ema
        bearish_trend = not price_above_1d_sma and not price_above_1d_ema
        
        # === RSI(2) EXTREME LEVELS ===
        rsi2_oversold = rsi_2[i] < 20  # Extreme oversold
        rsi2_overbought = rsi_2[i] > 80  # Extreme overbought
        
        # RSI(14) confirmation - confirm momentum shift
        rsi14_neutral = 30 < rsi_14[i] < 70
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: RSI(2) extreme oversold + bullish 1d trend + volume
        if rsi2_oversold and bullish_trend:
            if vol_above_avg:
                desired_signal = SIZE
            elif rsi14_neutral:
                desired_signal = SIZE * 0.5  # Half size without volume
        
        # SHORT: RSI(2) extreme overbought + bearish 1d trend + volume
        if rsi2_overbought and bearish_trend:
            if vol_above_avg:
                desired_signal = -SIZE
            elif rsi14_neutral:
                desired_signal = -SIZE * 0.5  # Half size without volume
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            stop_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        # === TAKE PROFIT (3R or opposite extreme) ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            stop_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        # === TRAILING STOP ACTIVATION (after 1.5R profit) ===
        if in_position and position_side > 0:
            if close[i] > entry_price + 1.5 * entry_atr:
                # Lock in profits - reduce to half
                if desired_signal == SIZE:
                    desired_signal = SIZE * 0.5
        
        if in_position and position_side < 0:
            if close[i] < entry_price - 1.5 * entry_atr:
                # Lock in profits - reduce to half
                if desired_signal == -SIZE:
                    desired_signal = -SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals