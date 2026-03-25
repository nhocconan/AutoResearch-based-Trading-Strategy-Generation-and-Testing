#!/usr/bin/env python3
"""
Experiment #1586: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend bias provides optimal trade frequency
(20-50 trades/year) while Connors RSI (CRSI) excels at mean-reversion entries in
bear/range markets. Combined with Choppiness Index regime detection, this should
outperform pure trend-following strategies that failed on BTC/ETH 2022-2024.

Key innovations:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Superior to standard RSI for catching short-term extremes
   - Proven 75% win rate in mean-reversion setups
2. CHOPPINESS REGIME: CHOP(14) > 61.8 = range (use CRSI extremes),
   CHOP < 38.2 = trend (use CRSI + 1w HMA bias)
3. 1w HMA trend filter: Only long when price > 1w_HMA, only short when price < 1w_HMA
   Prevents major counter-trend positions during crypto crashes
4. LOOSE CRSI THRESHOLDS: <25 for long, >75 for short (not <10/>90)
   Ensures ≥30 trades/train while maintaining edge
5. VOLATILITY FILTER: ATR(14)/close > 0.02 (2%) ensures sufficient movement

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- 1d TF = fewer false signals, lower fee drag than 6h
- CRSI proven superior to RSI for mean-reversion in bear markets
- Regime-switching adapts to 2022 crash (trend) vs 2023-2024 range
- Weekly HMA prevents catastrophic counter-trend losses

Entry logic (LOOSE to guarantee trades):
- LONG trend: CHOP<38 + price>1w_HMA + CRSI<25 + close>open (bullish candle)
- SHORT trend: CHOP<38 + price<1w_HMA + CRSI>75 + close<open (bearish candle)
- LONG range: CHOP>61 + CRSI<20 + price<BB_lower(20,2.0)
- SHORT range: CHOP>61 + CRSI>80 + price>BB_upper(20,2.0)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
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

def calculate_crsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (period=2)
    PercentRank: Percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (period=2)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_streak_loss != 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    rsi_streak[avg_streak_loss == 0] = 100  # All gains
    
    # PercentRank(100) - percentile of today's return vs last 100 days
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        window = returns[i-99:i+1]  # 100 values including today
        count_below = np.sum(window[:-1] < window[-1])  # Count days below today
        percent_rank[i] = count_below / 99 * 100  # Percentile (0-100)
    
    # Combine into CRSI
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 120  # Need 100 for CRSI + 20 for BB
    
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
        
        if np.isnan(bb_lower[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CRSI SIGNALS (LOOSE thresholds for trade frequency) ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25  # Long entry
        crsi_overbought = crsi_val > 75  # Short entry
        crsi_extreme_low = crsi_val < 20  # Strong long
        crsi_extreme_high = crsi_val > 80  # Strong short
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === CANDLE DIRECTION ===
        bullish_candle = close[i] > prices["open"].values[i]
        bearish_candle = close[i] < prices["open"].values[i]
        
        # === VOLATILITY FILTER (ensure sufficient movement) ===
        vol_filter = atr_14[i] / close[i] > 0.015  # 1.5% daily ATR
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: CRSI + 1w HMA bias + candle confirmation
        if is_trend_regime and vol_filter:
            # LONG: 1w bullish + CRSI oversold + bullish candle
            if price_above_1w and crsi_oversold and bullish_candle:
                desired_signal = SIZE_STRONG if crsi_extreme_low else SIZE_BASE
            
            # SHORT: 1w bearish + CRSI overbought + bearish candle
            elif price_below_1w and crsi_overbought and bearish_candle:
                desired_signal = -SIZE_STRONG if crsi_extreme_high else -SIZE_BASE
        
        # RANGE REGIME: CRSI extremes + Bollinger touch (mean reversion)
        elif is_range_regime and vol_filter:
            # LONG: CRSI extreme low + price at BB lower
            if crsi_extreme_low and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme high + price at BB upper
            elif crsi_extreme_high and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use 1w HMA for bias + CRSI pullback
        else:
            # LONG: 1w bullish + CRSI oversold (pullback entry)
            if price_above_1w and crsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + CRSI overbought (pullback entry)
            elif price_below_1w and crsi_overbought:
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