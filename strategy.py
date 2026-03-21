#!/usr/bin/env python3
"""
EXPERIMENT #007 - Supertrend + RSI Pullback with 4h Trend Filter (15m)
=======================================================================
Hypothesis: 15m Supertrend captures short-term momentum while 4h HMA(21)
ensures we trade with the major trend. RSI pullback entries (buy dips in
uptrend, sell rallies in downtrend) improve entry quality. ADX filter
avoids choppy markets. ATR trailing stop protects capital.

Key features:
- Primary TF: 15m (frequent signals, more trades than daily)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: Supertrend(10,3) flip + RSI(14) pullback confirmation
- Filter: ADX(14) > 20 for trend strength, volume > SMA(20)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels with take-profit scaling
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4h_filter_15m_v2"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, n):
        if trend[i-1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period-1] = tr[:period].mean()
    plus_di[period-1] = 100 * plus_dm[:period].sum() / (atr[period-1] * period + 1e-10)
    minus_di[period-1] = 100 * minus_dm[:period].sum() / (atr[period-1] * period + 1e-10)
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * (plus_dm[i] + (period - 1) * plus_di[i-1] / 100 * atr[i-1]) / (atr[i] * period + 1e-10)
        minus_di[i] = 100 * (minus_dm[i] + (period - 1) * minus_di[i-1] / 100 * atr[i-1]) / (atr[i] * period + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = 0.14  # Half position for take-profit
    
    # Track position state for stoploss and take-profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    take_profit_hit = False
    
    min_period = 50  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(supertrend[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # ADX filter - only trade when trend is strong enough
        adx_valid = adx[i] > 20
        
        # Volume filter - avoid low volume periods
        volume_valid = volume[i] > vol_sma[i] * 0.8
        
        # Supertrend signal
        st_signal = st_trend[i]  # 1 = bullish, -1 = bearish
        
        # RSI pullback logic
        rsi_pullback_valid = False
        if hma_trend == 1 and st_signal == 1:
            # Uptrend: buy on RSI pullback (RSI dipped but still > 40)
            rsi_pullback_valid = 40 < rsi[i] < 60
        elif hma_trend == -1 and st_signal == -1:
            # Downtrend: sell on RSI rally (RSI rose but still < 60)
            rsi_pullback_valid = 40 < rsi[i] < 60
        
        # Determine target signal
        target_signal = 0.0
        if adx_valid and volume_valid and rsi_pullback_valid:
            if hma_trend == 1 and st_signal == 1:
                target_signal = BASE_SIZE  # Long
            elif hma_trend == -1 and st_signal == -1:
                target_signal = -BASE_SIZE  # Short
        
        # Stoploss and take-profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Take profit at 2R
                if not take_profit_hit:
                    profit_target = entry_price + 2.0 * 2.5 * atr[i]  # 2R = 2 * stop distance
                    if close[i] >= profit_target:
                        take_profit_triggered = True
            else:
                # Short position
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Take profit at 2R
                if not take_profit_hit:
                    profit_target = entry_price - 2.0 * 2.5 * atr[i]
                    if close[i] <= profit_target:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            take_profit_hit = False
        elif take_profit_triggered:
            # Reduce to half position
            signals[i] = HALF_SIZE * position_side
            take_profit_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0:
                if position_side == 0:
                    # New entry
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    take_profit_hit = False
                elif np.sign(target_signal) == position_side:
                    # Same direction - maintain or increase
                    signals[i] = BASE_SIZE * position_side
                else:
                    # Reversal - close and open new
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    take_profit_hit = False
            elif position_side != 0:
                # Maintain existing position
                if take_profit_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = BASE_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals