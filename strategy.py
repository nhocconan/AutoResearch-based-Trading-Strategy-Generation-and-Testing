#!/usr/bin/env python3
"""
EXPERIMENT #031 - KAMA Adaptive Trend + Dual HTF Filter + ADX Regime (15m primary)
==================================================================================
Hypothesis: 15m KAMA adapts to volatility better than EMA/HMA, reducing whipsaws.
Dual HTF filter (4h HMA for major trend + 1h RSI for pullback timing) provides
stronger confluence than single HTF. ADX(14) > 25 ensures we only trade in
trending markets, avoiding chop that killed many 15m strategies.

Key features:
- Primary TF: 15m (MANDATORY for this experiment)
- HTF filter 1: 4h HMA(50) for major trend direction
- HTF filter 2: 1h RSI(14) for pullback entry timing (30-70 zone)
- Primary trend: KAMA(14) adaptive moving average
- Regime filter: ADX(14) > 25 (trending market only)
- Volume confirmation: volume > 1.5 * 20-bar avg volume
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.25 discrete levels (conservative for 15m noise)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this differs from failed 15m strategies:
- #025 hma_rsi_volume_zscore_15m_1h_v1 failed with DD=-75.9% (too aggressive sizing)
- This uses smaller size (0.25 vs likely 0.35+), dual HTF (not just 1h), ADX filter
- KAMA adapts faster than HMA in trending markets, slower in chop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_dual_htf_adx_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 indicates trending market
    """
    n = len(close)
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth with Wilder's method
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate DX and ADX
    plus_di = 100 * plus_di / (atr + 1e-10)
    minus_di = 100 * minus_di / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    kama = calculate_kama(close, 14)
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative position size for 15m noise
    HALF_SIZE = SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(vol_ma[i]) or atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF Filter 1: 4h HMA trend direction
        hma_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # HTF Filter 2: 1h RSI pullback zone (30-70 for entry timing)
        rsi_1h_value = rsi_1h_aligned[i]
        rsi_valid_long = 30 <= rsi_1h_value <= 70  # Not overbought for long
        rsi_valid_short = 30 <= rsi_1h_value <= 70  # Not oversold for short
        
        # Primary trend: KAMA direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # Regime filter: ADX > 25 (trending market)
        regime_valid = adx[i] > 25
        
        # Volume confirmation: volume > 1.5 * 20-bar avg
        volume_valid = volume[i] > 1.5 * vol_ma[i]
        
        # DI crossover confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All filters aligned
        if (hma_4h_trend == 1 and kama_trend == 1 and rsi_valid_long and
            regime_valid and volume_valid and di_bullish):
            target_signal = SIZE
        
        # Short entry: All filters aligned
        elif (hma_4h_trend == -1 and kama_trend == -1 and rsi_valid_short and
              regime_valid and volume_valid and di_bearish):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR)
                if not profit_target_hit and entry_atr > 0:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
                
                # Check trend reversal (KAMA flip)
                if kama_trend == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and entry_atr > 0:
                    if close[i] <= entry_price - 4.0 * entry_atr:
                        take_profit_triggered = True
                
                # Check trend reversal
                if kama_trend == 1:
                    trend_reversal = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 0.0
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        elif trend_reversal:
            # Exit on trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 0.0
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
                entry_atr = atr[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals