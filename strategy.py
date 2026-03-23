#!/usr/bin/env python3
"""
Experiment #288: 30m Primary + 4h/1d HTF — Simplified Trend Pullback

Hypothesis: Previous 30m strategies (#278, #280) failed with Sharpe=0.000 due to 
over-filtering (session + volume + CRSI all required = 0 trades).

This version uses SIMPLER entry logic for 30m:
- 4h HMA(21/50) for PRIMARY trend direction (most important)
- 1d HMA(50) for MACRO bias (soft filter only, not hard requirement)
- 30m RSI(7) pullback entries (35-65 zone - triggers more frequently than RSI14)
- 30m ATR(14) 2.5x trailing stoploss
- Position size: 0.20 (conservative for 30m volatility)
- NO session filter (killed #278 trades)
- NO volume filter (too strict)

KEY CHANGES from failed #278:
- REMOVED session filter entirely (was killing all trades)
- REMOVED volume filter entirely
- REMOVED Choppiness Index (adds complexity, minimal edge on 30m)
- REMOVED CRSI (too many conditions = 0 trades)
- RSI(7) 35-65 triggers ~35% of bars vs CRSI at ~10%
- 1d HMA is soft bias only, not hard requirement
- Smaller position size (0.20 vs 0.25) for 30m volatility

TARGET: 40-80 trades/year on 30m, Sharpe > 0.3 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_atr_v1"
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
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 30m entries
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_21_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50_raw = calculate_hma(df_4h['close'].values, 50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21_raw)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.20  # Conservative for 30m volatility
    
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
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) - SOFT FILTER ONLY ===
        # Not a hard requirement, just adds confidence
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) - PRIMARY FILTER ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 4h TREND SLOPE (price vs HMA21) ===
        price_above_hma_4h = close[i] > hma_4h_21_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_21_aligned[i]
        
        # === RSI PULLBACK SIGNALS (35-65 zone for 30m - triggers frequently) ===
        # Long: RSI pulled back but not oversold (35-65)
        rsi_pullback_long = (rsi_7[i] >= 35.0) and (rsi_7[i] <= 65.0)
        # Short: RSI rallied but not overbought (35-65)
        rsi_pullback_short = (rsi_7[i] >= 35.0) and (rsi_7[i] <= 65.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + price above 4h HMA21 + RSI pullback
        # 1d bias is soft (adds confidence but not required)
        if hma_4h_bullish and price_above_hma_4h and rsi_pullback_long:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h bearish + price below 4h HMA21 + RSI pullback
        elif hma_4h_bearish and price_below_hma_4h and rsi_pullback_short:
            desired_signal = -POSITION_SIZE
        
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
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_7[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_7[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
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