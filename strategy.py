#!/usr/bin/env python3
"""
EXPERIMENT #006 - Daily Supertrend + Weekly HMA Trend Filter Strategy (1d)
===========================================================================
Hypothesis: Daily timeframe captures multi-week swings with fewer whipsaws than intraday.
Weekly HMA provides robust major trend filter (institutional level), while daily Supertrend
gives clear entry/exit signals. RSI confirms pullback entries to avoid chasing tops/bottoms.

Key features:
- 1d primary timeframe (this experiment's rotation)
- 1w HMA for major trend filter (highest TF reliability)
- Supertrend(10, 3) for clear trend direction and stoploss levels
- RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend)
- ATR(14) for volatility-adjusted position sizing
- Conservative position size (0.25 entry, 0.15 half) for drawdown control
- Fixed discrete signal levels to minimize fee churn

Primary TF: 1d | HTF: 1w HMA trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_weekly_hma_1d_v1"
timeframe = "1d"
leverage = 1.0


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Calculate Supertrend indicator
    Returns: (supertrend_values, trend_direction)
    trend_direction: 1 = uptrend (price above supertrend), -1 = downtrend
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate final bands with trend logic
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    supertrend[period-1] = upper_band[period-1]
    trend[period-1] = -1  # Start in downtrend
    
    for i in range(period, n):
        if trend[i-1] == 1:
            # Previous trend was up
            if lower_band[i] < supertrend[i-1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] < supertrend[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
        else:
            # Previous trend was down
            if upper_band[i] > supertrend[i-1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] > supertrend[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
    
    # Set NaN for initial period
    supertrend[:period-1] = np.nan
    trend[:period-1] = np.nan
    
    return supertrend, trend


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_values = rsi.values.copy()
    rsi_values[:period + 1] = np.nan
    return rsi_values


def calculate_hma(close: np.ndarray, period: int = 21) -> np.ndarray:
    """Calculate Hull Moving Average (HMA)"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half_period, min_periods=half_period, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull_input = 2 * wma_half - wma_full
    hma = hull_input.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    hma_values = hma.values.copy()
    hma_values[:period] = np.nan
    return hma_values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values.copy()
    atr[:period] = np.nan
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data - use .copy() to avoid read-only issues
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    # Use 1w HMA as major trend filter
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values.copy(), period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)  # auto shift(1)
    
    # === CALCULATE 1d INDICATORS (vectorized before loop) ===
    supertrend, supertrend_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 1d HMA for additional confirmation
    hma_1d = calculate_hma(close, period=21)
    
    # === SIGNAL PARAMETERS ===
    SIZE_ENTRY = 0.25      # 25% position on entry
    SIZE_HALF = 0.15       # 15% after take profit
    STOPLOSS_MULT = 2.5    # 2.5*ATR stoploss (wider for daily)
    TAKEPROFIT_MULT = 2.0  # 2R take profit
    RSI_OVERSOLD = 40      # RSI < 40 for long entry (pullback)
    RSI_OVERBOUGHT = 60    # RSI > 60 for short entry (rally)
    
    signals = np.zeros(n)
    
    # Track position state for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    min_lookback = 50  # Ensure all indicators are valid (weekly alignment needs more)
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(supertrend_direction[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(hma_1d[i])):
            signals[i] = 0.0
            continue
        
        current_atr = atr[i]
        current_price = close[i]
        current_rsi = rsi[i]
        current_hma_1w = hma_1w_aligned[i]
        current_hma_1d = hma_1d[i]
        current_supertrend = supertrend[i]
        current_trend = supertrend_direction[i]
        
        # === MAJOR TREND FILTER (1w HMA) ===
        # Price above weekly HMA = major uptrend, below = major downtrend
        major_trend_up = current_price > current_hma_1w
        major_trend_down = current_price < current_hma_1w
        
        # === DAILY TREND (Supertrend) ===
        supertrend_up = current_trend == 1
        supertrend_down = current_trend == -1
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if position_side == 0:
            # === LONG ENTRY: weekly uptrend + supertrend up + RSI pullback ===
            if (major_trend_up and supertrend_up and current_rsi < RSI_OVERSOLD):
                new_signal = SIZE_ENTRY
                entry_price = current_price
                position_side = 1
                highest_since_entry = current_price
                lowest_since_entry = current_price
                trailing_stop = current_supertrend  # Use supertrend as initial stop
            
            # === SHORT ENTRY: weekly downtrend + supertrend down + RSI rally ===
            elif (major_trend_down and supertrend_down and current_rsi > RSI_OVERBOUGHT):
                new_signal = -SIZE_ENTRY
                entry_price = current_price
                position_side = -1
                highest_since_entry = current_price
                lowest_since_entry = current_price
                trailing_stop = current_supertrend  # Use supertrend as initial stop
        
        elif position_side == 1:
            # Track highest price since entry for trailing
            highest_since_entry = max(highest_since_entry, current_price)
            
            # === TRAILING STOP: use supertrend level ===
            if current_supertrend > trailing_stop:
                trailing_stop = current_supertrend
            
            # === STOPLOSS: price drops below supertrend or trailing stop ===
            if current_price < trailing_stop or current_trend == -1:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0 and current_atr > 0:
                profit_r = (current_price - entry_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price + 0.5 * current_atr
            
            # === EXIT: Major trend reversal (1w HMA) ===
            elif major_trend_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: RSI overbought in long position ===
            elif current_rsi > 75:
                new_signal = SIZE_HALF
            
            else:
                # Maintain position
                new_signal = SIZE_ENTRY if new_signal == 0 else new_signal
        
        elif position_side == -1:
            # Track lowest price since entry for trailing
            lowest_since_entry = min(lowest_since_entry, current_price)
            
            # === TRAILING STOP: use supertrend level ===
            if trailing_stop == 0 or current_supertrend < trailing_stop:
                trailing_stop = current_supertrend
            
            # === STOPLOSS: price rises above supertrend or trailing stop ===
            if current_price > trailing_stop or current_trend == 1:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0 and current_atr > 0:
                profit_r = (entry_price - current_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = -SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price - 0.5 * current_atr
            
            # === EXIT: Major trend reversal (1w HMA) ===
            elif major_trend_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: RSI oversold in short position ===
            elif current_rsi < 25:
                new_signal = -SIZE_HALF
            
            else:
                # Maintain position
                new_signal = -SIZE_ENTRY if new_signal == 0 else new_signal
        
        signals[i] = new_signal
    
    return signals