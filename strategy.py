#!/usr/bin/env python3
"""
Experiment #338: 30m Primary + 4h/1d HTF — Pullback Trend Following

Hypothesis: Previous 30m/1h failures (#328, #330, #335) had 0 trades from over-filtering
(session + volume + multiple HTF filters). This strategy uses SIMPLER logic:
1. 4h HMA(21) as PRIMARY trend filter (direction only)
2. 30m RSI(7) for pullback entries WITHIN the 4h trend (not extremes)
3. 30m BB Width for volatility regime (avoid low vol chop)
4. NO session/volume filters that killed trade count
5. Tighter stops (2*ATR) for lower TF, smaller size (0.20 base)

KEY INSIGHT: Lower TF needs FEWER filters, not more. Use 4h for direction,
30m for pullback timing. RSI 30-50 in uptrend = buy dip. RSI 50-70 in downtrend = sell rally.
Position size 0.20 (smaller for 30m fee impact).

TARGET: 40-80 trades/year on 30m, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_pullback_rsi_4h_trend_bb_v1"
timeframe = "30m"
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma
    return upper, lower, width

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile over lookback period."""
    bb_width_s = pd.Series(bb_width)
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    return percentile.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Smaller for 30m to reduce fee impact
    MAX_SIZE = 0.30
    
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
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        
        # Size modifier: aligned with 1d = full size, against 1d = reduced
        size_modifier = 1.0
        if price_above_hma_4h and not price_above_hma_1d:
            size_modifier = 0.75  # Long against 1d trend
        elif price_below_hma_4h and price_above_hma_1d:
            size_modifier = 0.75  # Short against 1d trend
        
        POSITION_SIZE = BASE_SIZE * size_modifier
        
        # === VOLATILITY REGIME (BB Width Percentile) ===
        # Avoid entering when BB width is at extreme lows (squeeze about to break)
        # or extreme highs (volatility spike, wait for calm)
        vol_ok = 20.0 < bb_width_pct[i] < 85.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 4h uptrend + 30m RSI pullback (not oversold, just cooling)
        if price_above_hma_4h and vol_ok:
            # RSI 7 between 35-55 = healthy pullback in uptrend
            if 35.0 < rsi_7[i] < 55.0:
                desired_signal = POSITION_SIZE
            # RSI 7 < 30 = oversold bounce opportunity
            elif rsi_7[i] < 30.0 and rsi_14[i] > 40.0:
                desired_signal = MAX_SIZE * size_modifier
        
        # SHORT: 4h downtrend + 30m RSI rally (not overbought, just bouncing)
        elif price_below_hma_4h and vol_ok:
            # RSI 7 between 45-65 = healthy rally in downtrend
            if 45.0 < rsi_7[i] < 65.0:
                desired_signal = -POSITION_SIZE
            # RSI 7 > 70 = overbought rejection opportunity
            elif rsi_7[i] > 70.0 and rsi_14[i] < 60.0:
                desired_signal = -MAX_SIZE * size_modifier
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_7[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_7[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend still valid
            if position_side > 0 and price_above_hma_4h:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -POSITION_SIZE
            else:
                # Trend reversed, exit
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
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