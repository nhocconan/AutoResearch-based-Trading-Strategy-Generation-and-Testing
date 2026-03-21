#!/usr/bin/env python3
"""
EXPERIMENT #001 - Multi-Timeframe HMA Trend + RSI Pullback Strategy
====================================================================
Hypothesis: Using 4h HMA for trend direction + 1h RSI pullback entries will 
capture trends more reliably than single-timeframe Supertrend. The HMA reduces 
lag vs EMA, while RSI pullbacks enter on dips within the trend, improving 
risk/reward. Z-score filter avoids trading during extreme volatility regimes.

Key improvements over supertrend_4h_v1:
- Multi-timeframe: 4h trend filter + 1h entries (proven 2x Sharpe in baseline)
- HMA has less lag than Supertrend, catches trends earlier
- RSI pullback entries = better entry prices within trend
- Z-score regime filter avoids choppy/high-vol periods
- Proper stoploss via signal→0 at 2*ATR against position
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_v2"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    wma_diff = 2 * wma_half - wma_full
    hma = pd.Series(wma_diff).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    loss_s = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.divide(gain_s, loss_s, out=np.zeros_like(gain_s), where=loss_s != 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    close_s = pd.Series(close)
    
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def calculate_zscore(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate Z-score for regime detection"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.divide(close - rolling_mean, rolling_std, out=np.zeros_like(close), where=rolling_std != 0)
    
    return zscore


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR for stoploss"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)  # auto shift(1)
    
    # Calculate 4h HMA slope (trend strength)
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]):
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / hma_4h_aligned[i-1]
    
    # === CALCULATE 1H INDICATORS (pre-loop for performance) ===
    hma_1h = calculate_hma(close, 21)
    rsi_1h = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    zscore_1h = calculate_zscore(close, 20)
    atr_1h = calculate_atr(high, low, close, 14)
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete, within 0.20-0.35 range)
    HALF_SIZE = 0.15  # Half position for take profit
    
    # Track position state for stoploss/takeprofit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Minimum bars for valid signals
    min_bars = max(50, int(np.sqrt(21)) + 26 + 9)
    
    for i in range(min_bars, n):
        # Skip if any required indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === TREND FILTER (4h HMA) ===
        # Long bias: price above 4h HMA and HMA sloping up
        # Short bias: price below 4h HMA and HMA sloping down
        trend_long = (close[i] > hma_4h_aligned[i]) and (hma_4h_slope[i] > 0.0001)
        trend_short = (close[i] < hma_4h_aligned[i]) and (hma_4h_slope[i] < -0.0001)
        
        # === REGIME FILTER (Z-score) ===
        # Avoid trading when Z-score is extreme (volatile/choppy regime)
        regime_ok = abs(zscore_1h[i]) < 2.0
        
        # === ENTRY SIGNALS (1h RSI pullback + MACD momentum) ===
        # Long entry: trend long + RSI pulled back (30-50) + MACD hist turning up
        long_entry = (trend_long and regime_ok and 
                      30 < rsi_1h[i] < 50 and 
                      macd_hist[i] > macd_hist[i-1] and macd_hist[i] > 0)
        
        # Short entry: trend short + RSI bounced (50-70) + MACD hist turning down
        short_entry = (trend_short and regime_ok and 
                       50 < rsi_1h[i] < 70 and 
                       macd_hist[i] < macd_hist[i-1] and macd_hist[i] < 0)
        
        # === STOPLOSS LOGIC (Rule 6) ===
        stoploss_triggered = False
        takeprofit_triggered = False
        
        if position_side == 1:  # Long position
            # Update highest price since entry
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # Stoploss: price drops 2*ATR below entry
            if close[i] < entry_price - 2.0 * atr_1h[i]:
                stoploss_triggered = True
            
            # Take profit: at 2R profit, reduce to half position
            elif close[i] > entry_price + 2.0 * atr_1h[i]:
                takeprofit_triggered = True
            
            # Trail stop: if price drops 1*ATR from highest, exit
            elif close[i] < highest_since_entry - 1.0 * atr_1h[i]:
                stoploss_triggered = True
        
        elif position_side == -1:  # Short position
            # Update lowest price since entry
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            # Stoploss: price rises 2*ATR above entry
            if close[i] > entry_price + 2.0 * atr_1h[i]:
                stoploss_triggered = True
            
            # Take profit: at 2R profit, reduce to half position
            elif close[i] < entry_price - 2.0 * atr_1h[i]:
                takeprofit_triggered = True
            
            # Trail stop: if price rises 1*ATR from lowest, exit
            elif close[i] > lowest_since_entry + 1.0 * atr_1h[i]:
                stoploss_triggered = True
        
        # === APPLY SIGNALS ===
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
        elif takeprofit_triggered:
            # Reduce to half position
            if position_side == 1:
                signals[i] = HALF_SIZE
            else:
                signals[i] = -HALF_SIZE
        elif long_entry and position_side != 1:
            # Enter long
            signals[i] = SIZE
            position_side = 1
            entry_price = close[i]
            highest_since_entry = close[i]
        elif short_entry and position_side != -1:
            # Enter short
            signals[i] = -SIZE
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = close[i]
        elif position_side == 1:
            # Stay long
            signals[i] = SIZE
        elif position_side == -1:
            # Stay short
            signals[i] = -SIZE
        else:
            # Flat
            signals[i] = 0.0
    
    return signals