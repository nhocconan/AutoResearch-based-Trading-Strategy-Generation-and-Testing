#!/usr/bin/env python3
"""
EXPERIMENT #043 - KAMA Adaptive Trend + RSI Pullback + ADX Regime (15m primary, 4h HTF)
=======================================================================================
Hypothesis: 15m timeframe provides more entry opportunities than 4h while 4h HTF KAMA
filters major trend direction. KAMA adapts to volatility (fast in trends, slow in chop).
ADX(14) > 20 ensures we only trade in trending markets. RSI(14) 35-65 pullback zone
times entries. Volume confirmation reduces false breakouts. This differs from failed
strategies by using adaptive KAMA instead of static EMA, proper 4h HTF alignment via
mtf_data helper, and conservative 0.25 position sizing with 2.5*ATR trailing stops.

Key features:
- Primary TF: 15m (this experiment's requirement)
- HTF filter: 4h KAMA(21) for major trend direction
- Trend: KAMA(14) on 15m + ADX(14) > 20 regime filter
- Entry: RSI(14) pullback to 35-65 zone in trend direction
- Volume: Current volume > 1.3x 20-period average
- Stoploss: 2.5*ATR(14) trailing, signal → 0 when hit
- Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
- Take profit: Reduce to half at 2R profit, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_adx_vol_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    KAMA adapts to market noise - fast during trends, slow during chop
    Based on Kaufman's "Trading Systems and Methods"
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        sum_volatility = 0.0
        for j in range(i - period + 1, i + 1):
            sum_volatility += abs(close[j] - close[j - 1])
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    sc[:] = np.nan
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) - Wilder's method"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Wilder's smoothing (EMA with alpha = 1/period)
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di_raw = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_di_raw = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_di_raw[i] / atr[i]
            minus_di[i] = 100 * minus_di_raw[i] / atr[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI - Wilder's method"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, 21)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)  # auto shift(1)
    
    # Calculate 15m indicators (pre-compute before loop - Rule 8)
    kama_15m = calculate_kama(close, 14)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Volume moving average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% of capital (conservative)
    STRONG_SIZE = 0.30  # 30% for strong signals
    HALF_SIZE = BASE_SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 60  # Wait for all indicators to stabilize (4h alignment + 15m indicators)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_15m[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_sma[i]) or atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter (major trend direction)
        htf_trend = 1 if close[i] > kama_4h_aligned[i] else -1
        
        # 15m KAMA trend (shorter-term direction)
        kama_trend = 1 if close[i] > kama_15m[i] else -1
        
        # ADX regime filter (must be > 20 for trending market)
        trend_strong = adx[i] > 20
        
        # RSI pullback zone (35-65 for entry timing)
        rsi_pullback_long = 35 <= rsi[i] <= 65
        rsi_pullback_short = 35 <= rsi[i] <= 65
        
        # Volume confirmation (> 1.3x average)
        volume_confirmed = volume[i] > 1.3 * vol_sma[i]
        
        # Determine target signal based on all filters
        target_signal = 0.0
        signal_strength = BASE_SIZE
        
        # Long entry: HTF bullish + 15m bullish + ADX strong + RSI pullback
        if (htf_trend == 1 and kama_trend == 1 and trend_strong and rsi_pullback_long):
            if volume_confirmed:
                signal_strength = STRONG_SIZE
            target_signal = signal_strength
        
        # Short entry: HTF bearish + 15m bearish + ADX strong + RSI pullback
        elif (htf_trend == -1 and kama_trend == -1 and trend_strong and rsi_pullback_short):
            if volume_confirmed:
                signal_strength = STRONG_SIZE
            target_signal = -signal_strength
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and kama_trend == -1:
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals