#!/usr/bin/env python3
"""
Experiment #1590: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: Connors RSI (CRSI) provides superior mean-reversion signals with 75% win rate
in literature. Combined with Choppiness Index regime detection and 4h HMA trend bias,
this should generate consistent trades in both bull and bear markets.

Key innovations vs failed 1h attempts:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI(14), catches short-term extremes
   - Long: CRSI<20 (loose, was <10 which gave 0 trades)
   - Short: CRSI>80 (loose, was >90 which gave 0 trades)
2. CHOPPINESS REGIME: CHOP<38 = trend (follow 4h HMA), CHOP>61 = range (mean revert)
3. SESSION FILTER: 06-22 UTC (not 08-20, too restrictive = 0 trades)
4. LOOSE THRESHOLDS: CRSI 20/80 (not 10/90), CHOP 38/61 (not 35/65)
5. 4h HMA(21) for trend bias - proven in mtf_hma_rsi_zscore_v1 (Sharpe=5.4)

Why this should beat failed 1h strategies (#1579, #1581, #1585, #1589 all Sharpe=0):
- LOOSE CRSI thresholds guarantee trades (20/80 vs 10/90)
- Wider session window (06-22 vs 08-20)
- Dual regime logic (trend + range) = more opportunities
- 4h HMA bias prevents major counter-trend trades

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG trend: 4h_HMA bullish + CHOP<38 + CRSI<20 + price>SMA50
- SHORT trend: 4h_HMA bearish + CHOP<38 + CRSI>80 + price<SMA50
- LONG range: CHOP>61 + CRSI<15 + price<BB_lower
- SHORT range: CHOP>61 + CRSI>85 + price>BB_upper

Target: Sharpe>0.6, trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.25 discrete (max 0.30)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_session_v2"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean-reversion signals
    CRSI = (RSI(period) + RSI_Streak(period) + PercentRank(period)) / 3
    
    Proven 75% win rate in literature for short-term reversals
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # Component 1: RSI(period)
    rsi_vals = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (absolute streak strength)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = np.abs(streak[i - streak_period + 1:i + 1])
        avg_streak = np.mean(streak_window)
        # Normalize to 0-100 scale
        streak_rsi[i] = min(100, max(0, avg_streak * 50))
    
    # Component 3: PercentRank of price change over rank_period
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period:i + 1])
        current_return = returns[-1] if len(returns) > 0 else 0
        rank = np.sum(returns <= current_return)
        pct_rank[i] = 100.0 * rank / len(returns) if len(returns) > 0 else 50
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_vals[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_vals[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def is_session_active(open_time, start_hour=6, end_hour=22):
    """
    Check if bar is within active trading session
    Default: 06-22 UTC (wider than 08-20 to ensure trades)
    """
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (06-22 UTC for liquidity) ===
        session_active = is_session_active(open_time[i], start_hour=6, end_hour=22)
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (4h and 1d HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_sma50 = close[i] > sma_50[i]
        price_below_sma50 = close[i] < sma_50[i]
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 20  # Loose: was <10 (0 trades)
        crsi_overbought = crsi_val > 80  # Loose: was >90 (0 trades)
        crsi_extreme_low = crsi_val < 15
        crsi_extreme_high = crsi_val > 85
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # Only trade during active session
        if session_active:
            # TREND REGIME: Follow 4h HMA direction + CRSI pullback
            if is_trend_regime:
                # LONG: 4h bullish + CRSI oversold + price above SMA50
                if price_above_4h and crsi_oversold and price_above_sma50:
                    desired_signal = SIZE_STRONG if price_above_1d else SIZE_BASE
                
                # SHORT: 4h bearish + CRSI overbought + price below SMA50
                elif price_below_4h and crsi_overbought and price_below_sma50:
                    desired_signal = -SIZE_STRONG if price_below_1d else -SIZE_BASE
            
            # RANGE REGIME: Mean reversion at Bollinger extremes
            elif is_range_regime:
                # LONG: CRSI extreme low + price at BB lower
                if crsi_extreme_low and bb_touch_lower:
                    desired_signal = SIZE_BASE
                
                # SHORT: CRSI extreme high + price at BB upper
                elif crsi_extreme_high and bb_touch_upper:
                    desired_signal = -SIZE_BASE
            
            # NEUTRAL REGIME: Use 1d HMA for bias + CRSI extremes
            else:
                # LONG: 1d bullish + CRSI oversold
                if price_above_1d and crsi_oversold:
                    desired_signal = SIZE_BASE
                
                # SHORT: 1d bearish + CRSI overbought
                elif price_below_1d and crsi_overbought:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals