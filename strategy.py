#!/usr/bin/env python3
"""
Experiment #392: 30m KAMA Adaptive Trend + 4h KAMA Filter + ADX + BB Regime

Hypothesis: After 391 failed experiments, the pattern is clear - static indicators
(EMA, HMA, RSI) fail because they don't adapt to changing volatility regimes.
KAMA (Kaufman Adaptive Moving Average) adjusts its smoothing based on market efficiency.

KEY INSIGHTS FROM FAILURES:
- Experiments 380-391 all have negative Sharpe (most with 0 trades!)
- CRSI strategies keep failing (too many filters = no trades)
- 1d HTF is too slow for 30m entries (lag causes missed moves)
- ADX>25 threshold too strict (rarely triggers)

NEW APPROACH FOR 30m:
1. KAMA adapts smoothing based on price efficiency ratio (ER)
   - High ER (trending) = fast KAMA (less smoothing)
   - Low ER (chop) = slow KAMA (more smoothing, fewer false signals)
2. 4h KAMA as trend filter (faster than 1d, better for 30m entries)
3. ADX>20 (not 25) for trend confirmation - generates MORE trades
4. Bollinger Band Width percentile for regime detection
   - BB Width < 20th percentile = squeeze (wait for breakout)
   - BB Width > 50th percentile = normal volatility (trade signals)
5. ATR stoploss at 2.0x (tighter than 2.5x to protect capital)
6. Position size 0.30 discrete (balance between return and DD)

Why 30m specifically:
- Captures intraday swings that 4h/1d miss
- Less noise than 5m/15m (fewer false signals)
- 4h HTF provides stable trend bias without excessive lag

Entry conditions (LOOSE to ensure trades):
- Long: price > 4h KAMA AND 30m KAMA > 30m EMA AND ADX > 20 AND BB normal
- Short: price < 4h KAMA AND 30m KAMA < 30m EMA AND ADX > 20 AND BB normal

Exit conditions:
- Signal flip (KAMA/EMA crossover against position)
- Stoploss (2.0 * ATR against entry)
- ADX drops below 18 (trend weakening - hysteresis)

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data helper
Position sizing: 0.30 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_adaptive_4h_kama_adx_bb_atr_v1"
timeframe = "30m"
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

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility using Efficiency Ratio (ER).
    ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    SC = (ER * (Fast SC - Slow SC) + Slow SC)^2
    KAMA = KAMA[prev] + SC * (Close - KAMA[prev])
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(smoothing_period, n):
        price_change = np.abs(close[i] - close[i - smoothing_period])
        noise = np.sum(np.abs(np.diff(close[i - smoothing_period:i + 1])))
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Calculate KAMA
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA of first smoothing_period bars
    kama[smoothing_period] = np.mean(close[:smoothing_period + 1])
    
    for i in range(smoothing_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 20 = trending market, ADX < 20 = ranging market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth TR, +DM, -DM using Wilder's method (ema with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate rolling percentile of BB Width for regime detection."""
    n = len(bandwidth)
    bb_pct = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(bandwidth[i]):
            continue
        window = bandwidth[i - lookback + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            bb_pct[i] = np.sum(valid_window < bandwidth[i]) / len(valid_window) * 100
    
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h KAMA for trend bias
    kama_4h = calculate_kama(df_4h['close'].values, fast_period=2, slow_period=30, smoothing_period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators
    kama_30m = calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10)
    ema_30m = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_30m[i]) or np.isnan(ema_30m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(bb_pct[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS FROM 4h KAMA ===
        bull_trend_4h = close[i] > kama_4h_aligned[i]
        bear_trend_4h = close[i] < kama_4h_aligned[i]
        
        # === 30m MOMENTUM (KAMA vs EMA) ===
        kama_above_ema = kama_30m[i] > ema_30m[i]
        kama_below_ema = kama_30m[i] < ema_30m[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 20  # Looser threshold for more trades
        trend_weak = adx[i] < 18    # Hysteresis for exit
        
        # === VOLATILITY REGIME (BB Width Percentile) ===
        bb_squeeze = bb_pct[i] < 20  # Very low volatility - wait
        bb_normal = bb_pct[i] >= 20  # Normal volatility - trade signals
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bull + 30m momentum + ADX strong + BB normal
        if bull_trend_4h and kama_above_ema and trend_strong and bb_normal:
            new_signal = SIZE
        
        # SHORT ENTRY: 4h bear + 30m momentum + ADX strong + BB normal
        elif bear_trend_4h and kama_below_ema and trend_strong and bb_normal:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND WEAKENING EXIT (ADX hysteresis) ===
        if in_position and new_signal != 0.0 and trend_weak:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === MOMENTUM REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_below_ema:
                new_signal = 0.0
            if position_side < 0 and kama_above_ema:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals