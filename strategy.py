#!/usr/bin/env python3
"""
Experiment #355: 6h Primary + 12h/1d HTF — BB-Keltner Squeeze Breakout v1

Hypothesis: Previous 6h strategies failed due to overly complex regime detection
(ADX+CHOP) that created conflicting signals. This version uses a PROVEN volatility
squeeze pattern (BB inside Keltner) which reliably precedes breakouts.

Key innovations:
1. BB-Keltner Squeeze: When BB bands are INSIDE Keltner channels = volatility compression
   → breakout imminent. This is a classic pattern from John Carter's "TTM Squeeze"
2. 12h HMA for intermediate trend direction (faster than 1d, slower than 6h)
3. 1d HMA for major trend bias (only trade in direction of 1d trend)
4. RSI(7) for momentum confirmation (faster than RSI(14) for 6h TF)
5. Volume spike confirmation on breakout (1.5x 20-bar avg)

Entry Logic:
- Squeeze Long: BB inside Keltner + 12h HMA bull + 1d HMA bull + breakout above BB upper
- Squeeze Short: BB inside Keltner + 12h HMA bear + 1d HMA bear + breakdown below BB lower
- RSI filter: RSI > 50 for longs, RSI < 50 for shorts (momentum confirmation)

Position sizing: 0.25 base, 0.30 when both 12h+1d aligned
Stoploss: 2.5x ATR(14) from entry price
Take profit: Reduce to half at 2R, trail stop

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=3 test, ALL symbols positive
Timeframe: 6h (30-60 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_keltner_squeeze_breakout_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def calculate_keltner_channels(high, low, close, period=20, atr_period=14, multiplier=1.5):
    """Keltner Channels - ATR-based envelope"""
    n = len(close)
    if n < period + atr_period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # EMA for center line
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # ATR for channel width
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, ema, lower

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    keltner_upper, keltner_mid, keltner_lower = calculate_keltner_channels(high, low, close, period=20, atr_period=14, multiplier=1.5)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 6h
    vol_sma = calculate_volume_sma(volume, 20)
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    take_profit_triggered = False
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(keltner_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # Squeeze = BB bands INSIDE Keltner channels
        # This indicates volatility compression → breakout imminent
        squeeze_long = bb_upper[i] < keltner_upper[i] and bb_lower[i] > keltner_lower[i]
        squeeze_active = squeeze_long
        
        # === HTF TREND BIAS ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === BREAKOUT DETECTION ===
        # Long breakout: price crosses above BB upper
        # Short breakout: price crosses below BB lower
        breakout_long = False
        breakout_short = False
        
        if i > 0 and not np.isnan(bb_upper[i-1]) and not np.isnan(bb_lower[i-1]):
            if close[i-1] <= bb_upper[i-1] and close[i] > bb_upper[i]:
                breakout_long = True
            if close[i-1] >= bb_lower[i-1] and close[i] < bb_lower[i]:
                breakout_short = True
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # === RSI MOMENTUM FILTER ===
        rsi_bull = rsi[i] > 50.0
        rsi_bear = rsi[i] < 50.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Squeeze + 12h bull + 1d bull + breakout + RSI bull
        if squeeze_active and htf_12h_bull and htf_1d_bull:
            if breakout_long and rsi_bull:
                desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
        
        # SHORT: Squeeze + 12h bear + 1d bear + breakout + RSI bear
        elif squeeze_active and htf_12h_bear and htf_1d_bear:
            if breakout_short and rsi_bear:
                desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        if in_position and not take_profit_triggered:
            if position_side > 0:
                profit_target = entry_price + 2.0 * 2.5 * entry_atr  # 2R = 2 * 2.5 ATR
                if high[i] >= profit_target:
                    take_profit_triggered = True
                    if desired_signal == 0.0:
                        desired_signal = SIZE_BASE / 2  # Reduce to half
            elif position_side < 0:
                profit_target = entry_price - 2.0 * 2.5 * entry_atr
                if low[i] <= profit_target:
                    take_profit_triggered = True
                    if desired_signal == 0.0:
                        desired_signal = -SIZE_BASE / 2
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE / 2
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE / 2
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                take_profit_triggered = False
                # Set stoploss
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
                take_profit_triggered = False
        
        signals[i] = final_signal
    
    return signals