#!/usr/bin/env python3
"""
Experiment #253: 15m Trend-Pullback Strategy with 4h HMA + 1h Supertrend + RSI + ADX

Hypothesis: 15m timeframe captures intraday swings but needs STRONG HTF filters to avoid noise.
Using 4h HMA for macro trend bias + 1h Supertrend for intermediate trend + 15m RSI pullback
entries + ADX filter to ensure actual trending conditions.

Why this might work on 15m:
- 4h HMA provides strong macro bias (only trade with HTF trend)
- 1h Supertrend confirms intermediate trend direction
- RSI(7) pullback to 40-50 (long) or 50-60 (short) = enter on dips in uptrend
- ADX(14) > 20 filters out choppy/range conditions
- ATR(14) stoploss at 2.0x controls drawdown
- Conservative sizing (0.25) with discrete levels minimizes fee drag

Key improvements over failed 15m experiments:
- #241 (15m trend pullback): Sharpe=-3.471 - likely too many trades, weak HTF filter
- #247 (15m chop regime): Sharpe=-3.271 - regime filter alone not enough
- This uses DUAL HTF (4h + 1h) for stronger trend confirmation
- RSI pullback (not extreme) = more trades than RSI<20/>80 strategies
- ADX filter prevents entries in low-volatility chop

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.15 half (discrete levels)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_4h_hma_1h_supertrend_rsi_adx_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend values, direction (1=above/bullish, -1=below/bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / \
              pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / \
               pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    dx = 100 * np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    supertrend_1h, supertrend_dir_1h = calculate_supertrend(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        period=10, 
        multiplier=3.0
    )
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    supertrend_dir_1h_aligned = align_htf_to_ltf(prices, df_1h, supertrend_dir_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend_dir_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(adx_14[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = macro trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend = intermediate trend
        bull_trend_1h = supertrend_dir_1h_aligned[i] == 1
        bear_trend_1h = supertrend_dir_1h_aligned[i] == -1
        
        # Strong bias: both 4h and 1h agree
        strong_bull = bull_trend_4h and bull_trend_1h
        strong_bear = bear_trend_4h and bear_trend_1h
        
        # === TREND STRENGTH FILTER ===
        # ADX > 20 = actual trending market (not chop)
        is_trending = adx_14[i] > 20
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Pullback in uptrend ---
        # Conditions:
        # 1. Strong bullish HTF bias (4h HMA + 1h Supertrend)
        # 2. ADX > 20 (trending, not chop)
        # 3. RSI(7) pullback to 40-50 range (not oversold, just dip)
        # 4. Price > EMA21 (short-term bullish)
        if strong_bull and is_trending:
            if 40 <= rsi_7[i] <= 55 and close[i] > ema_21[i]:
                new_signal = SIZE_BASE
        
        # --- SHORT ENTRY: Pullback in downtrend ---
        # Conditions:
        # 1. Strong bearish HTF bias (4h HMA + 1h Supertrend)
        # 2. ADX > 20 (trending, not chop)
        # 3. RSI(7) pullback to 45-60 range (not overbought, just rally)
        # 4. Price < EMA21 (short-term bearish)
        if strong_bear and is_trending:
            if 45 <= rsi_7[i] <= 60 and close[i] < ema_21[i]:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals