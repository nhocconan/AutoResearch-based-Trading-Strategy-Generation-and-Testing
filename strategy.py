#!/usr/bin/env python3
"""
Experiment #1284: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI + HMA)

Hypothesis: 12h timeframe offers optimal trade frequency (20-50/year) with lower fee drag.
This strategy uses a dual-regime approach that adapts to market conditions:

1. CHOPPINESS INDEX (CHOP) detects regime: CHOP>61.8 = range, CHOP<38.2 = trend
2. TRENDING REGIME: HMA(21) crossover + 1d/1w bias + ROC momentum confirmation
3. RANGING REGIME: Connors RSI mean reversion (CRSI<10 long, CRSI>90 short)
4. ATR(14) 2.5x trailing stop for all positions
5. LOOSE entry thresholds to guarantee 20-50 trades/year

Why this should work:
- 12h = proven higher TF success (lower noise, fewer false signals)
- Dual regime = adapts to 2022 crash (trending) and 2025 bear/range (mean revert)
- Connors RSI = 75% win rate on mean reversion (research-backed)
- 1d/1w HTF bias = strong directional filter without over-filtering
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Entry logic (LOOSE to guarantee trades):
- TREND LONG: CHOP<38.2 + 12h_HMA rising + 1d_HMA bullish + ROC>3
- TREND SHORT: CHOP<38.2 + 12h_HMA falling + 1d_HMA bearish + ROC<-3
- RANGE LONG: CHOP>61.8 + CRSI<15 + price>SMA200
- RANGE SHORT: CHOP>61.8 + CRSI>85 + price<SMA200

Target: Sharpe>0.5, trades>=20 train, trades>=3 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_hma_1d1w_v1"
timeframe = "12h"
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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - detects trending vs ranging markets
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3)
    rsi3 = np.full(n, np.nan, dtype=np.float64)
    for i in range(rsi_period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i - rsi_period + 1, i + 1):
            change = close[j] - close[j - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            rsi3[i] = 100.0
        else:
            rs = gains / losses
            rsi3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak(2) - streak of consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period + 1, n):
        streak = 0
        for j in range(i, max(-1, i - 20), -1):
            if close[j] > close[j - 1]:
                streak += 1
            elif close[j] < close[j - 1]:
                streak -= 1
            else:
                break
        
        # Convert streak to RSI-like value
        if streak >= 0:
            streak_rsi[i] = min(100.0, streak * 25.0)
        else:
            streak_rsi[i] = max(0.0, 100.0 + streak * 25.0)
    
    # Percent Rank(100) - where current return ranks in last 100 days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        current_return = close[i] - close[i - 1]
        count_below = 0
        for j in range(i - rank_period, i):
            past_return = close[j] - close[j - 1]
            if past_return < current_return:
                count_below += 1
        percent_rank[i] = (count_below / rank_period) * 100.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, period=200)
    hma_21 = calculate_hma(close, period=21)
    
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
    min_bars = 250  # Need enough bars for CRSI(100) + SMA(200)
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trending = chop < 45.0  # Slightly relaxed from 38.2 for more trades
        is_ranging = chop > 55.0   # Slightly relaxed from 61.8 for more trades
        
        # === HTF BIAS (1d + 1w HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 12h HMA slope (compare to 3 bars ago)
        hma_slope = 0.0
        if i >= 3 and not np.isnan(hma_21[i-3]):
            hma_slope = hma_21[i] - hma_21[i-3]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # TRENDING REGIME - Trend Following
        if is_trending:
            # LONG: HMA rising + 1d bullish + ROC positive
            if hma_slope > 0 and price_above_1d and roc_10[i] > 2.0:
                if price_above_1w:
                    desired_signal = SIZE_STRONG  # Strong bias (both 1d and 1w)
                else:
                    desired_signal = SIZE_BASE  # Basic trend
            
            # SHORT: HMA falling + 1d bearish + ROC negative
            elif hma_slope < 0 and price_below_1d and roc_10[i] < -2.0:
                if price_below_1w:
                    desired_signal = -SIZE_STRONG  # Strong bias
                else:
                    desired_signal = -SIZE_BASE  # Basic trend
        
        # RANGING REGIME - Mean Reversion (Connors RSI)
        elif is_ranging:
            # LONG: CRSI oversold + price above long-term SMA
            if crsi[i] < 20.0 and close[i] > sma_200[i]:
                if crsi[i] < 12.0:
                    desired_signal = SIZE_STRONG  # Very oversold
                else:
                    desired_signal = SIZE_BASE  # Moderately oversold
            
            # SHORT: CRSI overbought + price below long-term SMA
            elif crsi[i] > 80.0 and close[i] < sma_200[i]:
                if crsi[i] > 88.0:
                    desired_signal = -SIZE_STRONG  # Very overbought
                else:
                    desired_signal = -SIZE_BASE  # Moderately overbought
        
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