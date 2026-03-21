#!/usr/bin/env python3
"""
EXPERIMENT #002 - MTF HMA Trend + RSI Pullback Strategy (30m)
=============================================================
Hypothesis: Using 4h HMA for trend direction + 30m RSI pullback entries will
capture trends with better entry timing than pure trend-following. The 4h HMA
filters out counter-trend trades, while RSI oversold/overbought on 30m provides
optimal entry points within the trend.

Key improvements over failed #001:
- 4h HMA trend filter (smoother than Supertrend, fewer whipsaws)
- RSI pullback entries (buy dips in uptrend, sell rallies in downtrend)
- Proper stoploss: signal→0 at 2*ATR against position
- Take profit: reduce to half at 2R, trail stop at 1R
- Smaller position size (0.30 max) for better drawdown control
- Volume confirmation filter to avoid low-liquidity entries

Primary TF: 30m | HTF: 4h HMA trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_pullback_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate volume SMA for confirmation"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # === CALCULATE 30m INDICATORS (vectorized before loop) ===
    hma_30m = calculate_hma(close, 21)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # === SIGNAL PARAMETERS ===
    SIZE_ENTRY = 0.30      # 30% position on entry
    SIZE_HALF = 0.15       # 15% after take profit
    STOPLOSS_MULT = 2.0    # 2*ATR stoploss
    TAKEPROFIT_MULT = 2.0  # 2R take profit
    
    signals = np.zeros(n)
    
    # Track position state for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    min_lookback = max(50, 21)  # Ensure all indicators are valid
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        current_atr = atr[i]
        current_price = close[i]
        current_rsi = rsi[i]
        current_hma_4h = hma_4h_aligned[i]
        current_hma_30m = hma_30m[i]
        current_vol = volume[i]
        avg_vol = vol_sma[i]
        
        # Volume filter: require at least 80% of average volume
        volume_ok = current_vol >= 0.8 * avg_vol if avg_vol > 0 else True
        
        # === TREND FILTER (4h HMA) ===
        # Price above 4h HMA = uptrend, below = downtrend
        trend_up = current_price > current_hma_4h
        trend_down = current_price < current_hma_4h
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if position_side == 0:
            # === LONG ENTRY: uptrend + RSI pullback (30-45) ===
            if trend_up and volume_ok and 30 <= current_rsi <= 45:
                # Confirm 30m HMA also supportive
                if current_price > current_hma_30m:
                    new_signal = SIZE_ENTRY
                    entry_price = current_price
                    position_side = 1
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
            
            # === SHORT ENTRY: downtrend + RSI rally (55-70) ===
            elif trend_down and volume_ok and 55 <= current_rsi <= 70:
                # Confirm 30m HMA also supportive
                if current_price < current_hma_30m:
                    new_signal = -SIZE_ENTRY
                    entry_price = current_price
                    position_side = -1
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
        
        elif position_side == 1:
            # Track highest price since entry for trailing
            highest_since_entry = max(highest_since_entry, current_price)
            
            # === STOPLOSS: price drops 2*ATR below entry ===
            stoploss_price = entry_price - STOPLOSS_MULT * current_atr
            if current_price < stoploss_price:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (current_price - entry_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT and new_signal == 0:
                    new_signal = SIZE_HALF
                    # Trail stop: move stoploss to entry + 1*ATR
                    # (handled by checking stoploss each bar)
                    entry_price = current_price - STOPLOSS_MULT * current_atr / 2  # Move stop up
            
            # === EXIT: RSI overbought in long position ===
            elif current_rsi > 75:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
            
            else:
                new_signal = SIZE_ENTRY if new_signal == 0 else new_signal
        
        elif position_side == -1:
            # Track lowest price since entry for trailing
            lowest_since_entry = min(lowest_since_entry, current_price)
            
            # === STOPLOSS: price rises 2*ATR above entry ===
            stoploss_price = entry_price + STOPLOSS_MULT * current_atr
            if current_price > stoploss_price:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (entry_price - current_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT and new_signal == 0:
                    new_signal = -SIZE_HALF
                    entry_price = current_price + STOPLOSS_MULT * current_atr / 2
            
            # === EXIT: RSI oversold in short position ===
            elif current_rsi < 25:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
            
            else:
                new_signal = -SIZE_ENTRY if new_signal == 0 else new_signal
        
        # === TREND REVERSAL EXIT ===
        if position_side == 1 and trend_down:
            new_signal = 0.0
            position_side = 0
            entry_price = 0.0
        elif position_side == -1 and trend_up:
            new_signal = 0.0
            position_side = 0
            entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals