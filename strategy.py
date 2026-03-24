#!/usr/bin/env python3
"""
Experiment #855: 6h Primary + 1d/1w HTF — VW-RSI + Dual HMA + Vol-Adaptive Regime

Hypothesis: 6h timeframe with dual HTF confirmation (1w macro + 1d intermediate) 
provides optimal signal quality. Volume-Weighted RSI improves on standard RSI by 
incorporating volume conviction. ROC momentum filter confirms trend strength before 
entry. Volatility-adaptive Choppiness threshold adjusts regime detection to current 
market conditions (wider bands in high vol, tighter in low vol).

Key innovations:
1. 1w HMA(21) for macro trend bias - extremely slow, filters noise
2. 1d HMA(16) for intermediate trend confirmation
3. 6h VW-RSI(14) - volume-weighted RSI for better signal quality
4. ROC(10) momentum filter - only enter when momentum confirms direction
5. Vol-adaptive CHOP threshold - threshold scales with ATR percentile
6. Asymmetric entry/exit - easier entry (loose), stricter exit (reduce churn)
7. ATR(14) 2.5x trailing stop for risk management

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1w HMA bull + 1d HMA bull + VW-RSI<45 + ROC>0
- SHORT: 1w HMA bear + 1d HMA bear + VW-RSI>55 + ROC<0
- Regime filter relaxes thresholds in ranging markets

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vwrsi_dual_hma_roc_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        if span < 1:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_vw_rsi(close, volume, period=14):
    """
    Volume-Weighted RSI (VW-RSI)
    Uses volume-weighted gains/losses instead of simple price changes
    More reliable than standard RSI as it incorporates volume conviction
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta * volume, 0.0)
    loss = np.where(delta < 0, -delta * volume, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    vw_rsi = np.zeros(n)
    vw_rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            vw_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain[i] > 1e-10:
            vw_rsi[i] = 100.0
        else:
            vw_rsi[i] = 50.0
    
    return vw_rsi

def calculate_roc(close, period=10):
    """Rate of Change (Momentum)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.zeros(n)
    roc[:] = np.nan
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr_percentile(atr, lookback=50):
    """Calculate ATR percentile for vol-adaptive thresholds"""
    n = len(atr)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(atr[i]):
            window = atr[i - lookback:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                rank = np.sum(valid < atr[i])
                percentile[i] = rank / len(valid)
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=16)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h_16 = calculate_hma(close, period=16)
    hma_6h_48 = calculate_hma(close, period=48)
    vw_rsi = calculate_vw_rsi(close, volume, period=14)
    roc_10 = calculate_roc(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_pct = calculate_atr_percentile(atr_14, lookback=50)
    
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
    
    # Signal persistence to reduce churn
    prev_signal = 0.0
    signal_bar_count = 0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h_16[i]) or np.isnan(hma_6h_48[i]) or np.isnan(vw_rsi[i]) or np.isnan(roc_10[i]):
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
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bull = htf_1w_bull and htf_1d_bull
        htf_strong_bear = htf_1w_bear and htf_1d_bear
        htf_weak_bull = htf_1w_bull or htf_1d_bull
        htf_weak_bear = htf_1w_bear or htf_1d_bear
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_6h_16[i] > hma_6h_48[i]
        hma_6h_bear = hma_6h_16[i] < hma_6h_48[i]
        
        # === VW-RSI CONDITIONS (LOOSE for trades) ===
        vw_rsi_oversold = vw_rsi[i] < 45.0
        vw_rsi_overbought = vw_rsi[i] > 55.0
        vw_rsi_extreme_long = vw_rsi[i] < 30.0
        vw_rsi_extreme_short = vw_rsi[i] > 70.0
        
        # === ROC MOMENTUM ===
        roc_positive = roc_10[i] > 0.0
        roc_negative = roc_10[i] < 0.0
        roc_strong_long = roc_10[i] > 2.0
        roc_strong_short = roc_10[i] < -2.0
        
        # === CHOPPINESS REGIME (Vol-Adaptive Threshold) ===
        # In high vol (atr_pct > 0.7), use wider thresholds
        # In low vol (atr_pct < 0.3), use tighter thresholds
        vol_adj = 0.0
        if not np.isnan(atr_pct[i]):
            if atr_pct[i] > 0.7:
                vol_adj = 10.0  # High vol: wider bands
            elif atr_pct[i] < 0.3:
                vol_adj = -10.0  # Low vol: tighter bands
        
        chop_trending = chop_14[i] < (50.0 + vol_adj)
        chop_ranging = chop_14[i] >= (50.0 + vol_adj)
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_strong_bull:
            # Strong bullish HTF bias
            if chop_trending:
                # Trend regime: use HMA + ROC
                if hma_6h_bull and roc_positive:
                    if vw_rsi_oversold or vw_rsi[i] < 50:
                        if roc_strong_long:
                            desired_signal = SIZE_STRONG
                        else:
                            desired_signal = SIZE_BASE
            else:
                # Range regime: use VW-RSI mean reversion (loose)
                if vw_rsi_oversold:
                    if vw_rsi_extreme_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        elif htf_strong_bear:
            # Strong bearish HTF bias
            if chop_trending:
                # Trend regime: use HMA + ROC
                if hma_6h_bear and roc_negative:
                    if vw_rsi_overbought or vw_rsi[i] > 50:
                        if roc_strong_short:
                            desired_signal = -SIZE_STRONG
                        else:
                            desired_signal = -SIZE_BASE
            else:
                # Range regime: use VW-RSI mean reversion (loose)
                if vw_rsi_overbought:
                    if vw_rsi_extreme_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        elif htf_weak_bull and hma_6h_bull:
            # Weak bull + local bull = smaller position
            if vw_rsi_oversold and roc_positive:
                desired_signal = SIZE_BASE * 0.8
        
        elif htf_weak_bear and hma_6h_bear:
            # Weak bear + local bear = smaller position
            if vw_rsi_overbought and roc_negative:
                desired_signal = -SIZE_BASE * 0.8
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            # Keep small signals if we're already in position (reduce churn)
            if in_position and np.sign(desired_signal) == position_side:
                final_signal = position_side * SIZE_BASE
            else:
                final_signal = 0.0
        else:
            final_signal = 0.0
        
        # === SIGNAL PERSISTENCE (reduce churn) ===
        # Only flip signal if new signal persists for 2 bars OR strong reversal
        if final_signal != 0.0 and final_signal != prev_signal:
            if np.sign(final_signal) != np.sign(prev_signal) and prev_signal != 0.0:
                # Reversal: require stronger confirmation
                signal_bar_count += 1
                if signal_bar_count < 2:
                    final_signal = prev_signal  # Keep old signal
                else:
                    signal_bar_count = 0
            else:
                signal_bar_count = 0
        elif final_signal == 0.0 and prev_signal != 0.0:
            # Exit: allow immediately (stoploss or trend change)
            signal_bar_count = 0
        else:
            signal_bar_count = 0
        
        prev_signal = final_signal
        
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