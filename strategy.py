#!/usr/bin/env python3
"""
Experiment #270: 1d Adaptive Trend Following with Weekly HMA Bias and ADX Filter

Hypothesis: Daily timeframe offers cleaner trend signals with less noise than intraday.
After 269 experiments, the pattern shows:
- Simple trend-following with strong HTF bias works better than mean-reversion
- KAMA (Kaufman Adaptive) adapts to volatility better than EMA/SMA
- Weekly HMA provides stronger directional filter than daily
- ADX filter prevents entries during weak/choppy trends (critical for 2022)
- Looser entry thresholds needed on 1d to ensure >=10 trades per symbol

Key innovations:
1. KAMA(21) adaptive trend - responds faster in trends, slower in ranges
2. 1w HMA(21) bias - strongest HTF filter available (call ONCE before loop)
3. ADX(14) > 20 filter - only trade when trend has momentum
4. Price pullback to KAMA entry - better risk/reward than breakout
5. 2.5*ATR stoploss - appropriate for daily timeframe
6. Asymmetric sizing - reduce size when ADX weakens

Why 1d might work:
- Less noise = fewer false signals
- Lower fee drag (fewer trades)
- Better suited for multi-week trends
- 2022 crash had clear daily downtrend (ADX stayed elevated)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_trend_1w_hma_adx_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    More responsive in trends, smoother in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        if change == 0:
            er = 0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
            er = change / volatility if volatility > 0 else 0
        
        # Smoothing constant
        sc = (er * (2 / (fast_period + 1) - 2 / (slow_period + 1)) + 2 / (slow_period + 1)) ** 2
        
        # Initialize KAMA
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength regardless of direction.
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed averages (Wilder's method)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2 - 1:] = adx_raw[period * 2 - 1:]
    
    return adx

def calculate_ema(close, period=50):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 21)
    adx = calculate_adx(high, low, close, 14)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in weak trend
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = strongest directional bias (hard filter)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 20 = trend has momentum (avoid choppy markets)
        # ADX > 30 = strong trend (increase position size)
        trend_strong = adx[i] > 25
        trend_weak = adx[i] < 20
        
        # === KAMA TREND DIRECTION ===
        # Price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # KAMA slope (trend direction)
        kama_slope_bull = kama[i] > kama[i - 5] if i >= 5 else False
        kama_slope_bear = kama[i] < kama[i - 5] if i >= 5 else False
        
        # === DETERMINE POSITION SIZE ===
        if trend_strong:
            position_size = SIZE_BASE
        elif trend_weak:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        # LONG: 1w bias up + ADX confirms trend + price above KAMA + KAMA sloping up
        # Looser conditions to ensure >=10 trades per symbol on daily data
        long_conditions = (
            bull_trend_1w and  # 1w HMA bias bullish
            adx[i] > 20 and  # Trend has momentum
            price_above_kama and  # Price above adaptive MA
            kama_slope_bull  # KAMA sloping upward
        )
        
        # SHORT: Mirror of long
        short_conditions = (
            bear_trend_1w and  # 1w HMA bias bearish
            adx[i] > 20 and  # Trend has momentum
            price_below_kama and  # Price below adaptive MA
            kama_slope_bear  # KAMA sloping downward
        )
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position OR ADX drops too low
        if in_position and new_signal != 0.0:
            if position_side > 0 and (bear_trend_1w or adx[i] < 18):
                new_signal = 0.0  # 1w trend reversed or trend weakened
            if position_side < 0 and (bull_trend_1w or adx[i] < 18):
                new_signal = 0.0  # 1w trend reversed or trend weakened
        
        # === KAMA CROSS EXIT ===
        # Exit if price crosses KAMA against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and price_below_kama:
                new_signal = 0.0  # Price crossed below KAMA
            if position_side < 0 and price_above_kama:
                new_signal = 0.0  # Price crossed above KAMA
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals