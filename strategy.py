#!/usr/bin/env python3
"""
Experiment #1282: 4h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: Pure trend-following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear).
This strategy uses Choppiness Index to detect regime and switches logic accordingly:

1. RANGE REGIME (CHOP > 61.8): Mean reversion with Connors RSI
   - Long: CRSI < 10 + price > SMA200
   - Short: CRSI > 90 + price < SMA200
   - 75% win rate in research literature

2. TREND REGIME (CHOP < 38.2): Trend following with HMA
   - Long: HMA rising + price > HMA + weekly bias bullish
   - Short: HMA falling + price < HMA + weekly bias bearish

3. TRANSITION (38.2 <= CHOP <= 61.8): Flat or reduce position

Why this should work:
- Adapts to market conditions (range vs trend)
- Connors RSI proven for mean reversion in crypto
- 4h timeframe = 20-50 trades/year (fee-friendly)
- 1d Choppiness for regime, 1w HMA for major bias
- Discrete sizing minimizes fee churn

Entry conditions LOOSE enough to guarantee 30+ trades/year:
- CRSI thresholds: <15 / >85 (not extreme <10 / >90)
- CHOP thresholds: 55 / 65 (wider band for more regime switches)
- ROC confirmation: >2% / <-2% (not extreme)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_hma_1d1w_v1"
timeframe = "4h"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        if not np.isnan(streak[i]):
            # Map streak to 0-100 scale
            max_streak = max(abs(streak[i]), 1)
            if streak[i] > 0:
                streak_rsi[i] = 50.0 + min(streak[i] / max_streak * 50.0, 50.0)
            else:
                streak_rsi[i] = 50.0 - min(abs(streak[i]) / max_streak * 50.0, 50.0)
    
    # Percent Rank of returns
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            count_below = np.sum(valid < returns[i])
            percent_rank[i] = (count_below / len(valid)) * 100.0
    
    # Combine into Connors RSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Range/Ranging market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.isnan(atr[i]):
            continue
        
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_roc(close, period=10):
    """Rate of Change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    chop_1d_raw = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    roc_10 = calculate_roc(close, period=10)
    
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
    min_bars = 250  # Need enough bars for SMA200, CRSI rank_period, etc.
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(sma_200[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_1d_aligned[i]
        
        is_range_regime = chop > 55.0  # Slightly lower threshold for more range detection
        is_trend_regime = chop < 45.0  # Slightly higher threshold for more trend detection
        # 45-55 = transition zone (flat)
        
        # === WEEKLY BIAS (1w HMA) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === 4h TREND ===
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h[i-3]):
            hma_4h_slope = hma_4h[i] - hma_4h[i-3]
        
        price_above_hma = close[i] > hma_4h[i]
        price_below_hma = close[i] < hma_4h[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === MOMENTUM ===
        roc = roc_10[i] if not np.isnan(roc_10[i]) else 0.0
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_range_regime:
            # MEAN REVERSION with Connors RSI
            # Long: CRSI oversold + price above SMA200 (uptrend pullback)
            if crsi_val < 20.0 and price_above_sma200:
                if roc > -2.0:  # Not crashing
                    if crsi_val < 12.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: CRSI overbought + price below SMA200 (downtrend rally)
            elif crsi_val > 80.0 and price_below_sma200:
                if roc < 2.0:  # Not rallying hard
                    if crsi_val > 88.0:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        elif is_trend_regime:
            # TREND FOLLOWING with HMA
            # Long: HMA rising + price above HMA + weekly bullish
            if hma_4h_slope > 0 and price_above_hma and weekly_bullish:
                if roc > 2.0:  # Momentum confirmation
                    if roc > 6.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: HMA falling + price below HMA + weekly bearish
            elif hma_4h_slope < 0 and price_below_hma and weekly_bearish:
                if roc < -2.0:  # Momentum confirmation
                    if roc < -6.0:
                        desired_signal = -SIZE_STRONG
                    else:
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