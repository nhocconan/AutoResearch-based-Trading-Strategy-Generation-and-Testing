#!/usr/bin/env python3
"""
EXPERIMENT #007 - 15m Multi-Timeframe Momentum with 4h Trend Filter
====================================================================
Hypothesis: 15m timeframe captures intraday momentum swings while 4h HMA provides
robust trend filter. Combining 1h RSI pullback entries with 15m Supertrend exits
should reduce whipsaws compared to pure 15m strategies. Z-score filter avoids
chasing extreme moves.

Key features:
- 15m primary timeframe (this experiment's rotation)
- 4h HMA(21) for major trend filter (institutional level)
- 1h RSI(14) for pullback entry timing
- 15m Supertrend(10,3) for exit signals and trailing stops
- Z-score(20) filter to avoid extreme entries
- ATR(14) for volatility-adjusted stoploss
- Conservative position size (0.25 entry, 0.15 half) for drawdown control
- Discrete signal levels to minimize fee churn

Primary TF: 15m | HTF: 4h HMA trend + 1h RSI timing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_momentum_15m_4h_hma_v1"
timeframe = "15m"
leverage = 1.0


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
            if upper_band[i] > supertrend[i-1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] > supertrend[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
    
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


def calculate_zscore(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate Z-score (standardized price deviation from mean)"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.nan)
    
    zscore_values = zscore.values.copy()
    zscore_values[:period] = np.nan
    return zscore_values


def calculate_ema(close: np.ndarray, period: int = 21) -> np.ndarray:
    """Calculate EMA with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    ema_values = ema.values.copy()
    ema_values[:period] = np.nan
    return ema_values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data - use .copy() to avoid read-only issues
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    # 4h HMA for major trend filter
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values.copy(), period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # 1h RSI for entry timing
    df_1h = get_htf_data(prices, '1h')
    rsi_1h = calculate_rsi(df_1h['close'].values.copy(), period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)  # auto shift(1)
    
    # === CALCULATE 15m INDICATORS (vectorized before loop) ===
    supertrend, supertrend_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    atr = calculate_atr(high, low, close, period=14)
    zscore = calculate_zscore(close, period=20)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    # === SIGNAL PARAMETERS ===
    SIZE_ENTRY = 0.25      # 25% position on entry
    SIZE_HALF = 0.15       # 15% after take profit
    STOPLOSS_MULT = 2.0    # 2*ATR stoploss
    TAKEPROFIT_MULT = 2.0  # 2R take profit
    RSI_OVERSOLD = 45      # RSI < 45 for long entry (pullback)
    RSI_OVERBOUGHT = 55    # RSI > 55 for short entry (rally)
    ZSCORE_EXTREME = 2.0   # Avoid entries when |zscore| > 2
    
    signals = np.zeros(n)
    
    # Track position state for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    min_lookback = 100  # Ensure all indicators are valid
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or
            np.isnan(supertrend[i]) or np.isnan(supertrend_direction[i]) or 
            np.isnan(atr[i]) or np.isnan(zscore[i]) or 
            np.isnan(ema_21[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        current_atr = atr[i]
        current_price = close[i]
        current_hma_4h = hma_4h_aligned[i]
        current_rsi_1h = rsi_1h_aligned[i]
        current_zscore = zscore[i]
        current_supertrend = supertrend[i]
        current_trend = supertrend_direction[i]
        current_ema_21 = ema_21[i]
        current_ema_50 = ema_50[i]
        
        # === MAJOR TREND FILTER (4h HMA) ===
        major_trend_up = current_price > current_hma_4h
        major_trend_down = current_price < current_hma_4h
        
        # === 15m TREND (Supertrend + EMA) ===
        supertrend_up = current_trend == 1
        supertrend_down = current_trend == -1
        ema_bullish = current_ema_21 > current_ema_50
        ema_bearish = current_ema_21 < current_ema_50
        
        # === Z-SCORE FILTER (avoid extreme entries) ===
        zscore_ok_long = current_zscore < ZSCORE_EXTREME  # Not overbought
        zscore_ok_short = current_zscore > -ZSCORE_EXTREME  # Not oversold
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if position_side == 0:
            # === LONG ENTRY: 4h uptrend + 15m supertrend up + 1h RSI pullback + Z-score ok ===
            if (major_trend_up and supertrend_up and ema_bullish and 
                current_rsi_1h < RSI_OVERSOLD and zscore_ok_long):
                new_signal = SIZE_ENTRY
                entry_price = current_price
                position_side = 1
                highest_since_entry = current_price
                lowest_since_entry = current_price
                trailing_stop = current_price - STOPLOSS_MULT * current_atr
            
            # === SHORT ENTRY: 4h downtrend + 15m supertrend down + 1h RSI rally + Z-score ok ===
            elif (major_trend_down and supertrend_down and ema_bearish and 
                  current_rsi_1h > RSI_OVERBOUGHT and zscore_ok_short):
                new_signal = -SIZE_ENTRY
                entry_price = current_price
                position_side = -1
                highest_since_entry = current_price
                lowest_since_entry = current_price
                trailing_stop = current_price + STOPLOSS_MULT * current_atr
        
        elif position_side == 1:
            # Track highest price since entry for trailing
            highest_since_entry = max(highest_since_entry, current_price)
            
            # === UPDATE TRAILING STOP (lock in profits) ===
            profit_atr = (current_price - entry_price) / current_atr if current_atr > 0 else 0
            if profit_atr > 1.0:
                # Trail at 1R profit once we have 1R gain
                new_trailing = current_price - 1.0 * current_atr
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
            
            # === STOPLOSS: price drops below trailing stop or supertrend flips ===
            if current_price < trailing_stop or current_trend == -1:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif profit_atr >= TAKEPROFIT_MULT:
                new_signal = SIZE_HALF
                # Lock in at least 1R profit
                trailing_stop = max(trailing_stop, entry_price + 1.0 * current_atr)
            
            # === EXIT: Major trend reversal (4h HMA) ===
            elif major_trend_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: 1h RSI extremely overbought ===
            elif current_rsi_1h > 75:
                new_signal = SIZE_HALF
            
            else:
                # Maintain position
                new_signal = SIZE_ENTRY if new_signal == 0 else new_signal
        
        elif position_side == -1:
            # Track lowest price since entry for trailing
            lowest_since_entry = min(lowest_since_entry, current_price)
            
            # === UPDATE TRAILING STOP (lock in profits) ===
            profit_atr = (entry_price - current_price) / current_atr if current_atr > 0 else 0
            if profit_atr > 1.0:
                # Trail at 1R profit once we have 1R gain
                new_trailing = current_price + 1.0 * current_atr
                if trailing_stop == 0 or new_trailing < trailing_stop:
                    trailing_stop = new_trailing
            
            # === STOPLOSS: price rises above trailing stop or supertrend flips ===
            if current_price > trailing_stop or current_trend == 1:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif profit_atr >= TAKEPROFIT_MULT:
                new_signal = -SIZE_HALF
                # Lock in at least 1R profit
                trailing_stop = min(trailing_stop, entry_price - 1.0 * current_atr) if trailing_stop > 0 else entry_price - 1.0 * current_atr
            
            # === EXIT: Major trend reversal (4h HMA) ===
            elif major_trend_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: 1h RSI extremely oversold ===
            elif current_rsi_1h < 25:
                new_signal = -SIZE_HALF
            
            else:
                # Maintain position
                new_signal = -SIZE_ENTRY if new_signal == 0 else new_signal
        
        signals[i] = new_signal
    
    return signals