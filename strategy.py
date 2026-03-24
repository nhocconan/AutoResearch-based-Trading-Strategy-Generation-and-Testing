#!/usr/bin/env python3
"""
Experiment #1090: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + ADX Regime

Hypothesis: Fisher Transform catches reversals better than RSI in bear/range markets (2022-2025).
Combined with 4h HMA for trend direction and ADX for regime detection, this should generate
quality trades with proper frequency (40-80/year on 1h).

Key innovations:
1. Fisher Transform (period=9): Normalizes price to Gaussian distribution, extreme values (-2/+2)
   indicate reversal points. More responsive than RSI in choppy markets.
2. 4h HMA(21) for trend direction: Aligned properly using mtf_data helper
3. 1d HMA(21) for long-term bias filter: Only long if price>1d_HMA, only short if price<1d_HMA
4. ADX(14) regime filter: ADX>25 = trend (follow 4h HMA), ADX<20 = range (Fisher mean revert)
5. Session filter: Only trade 08-20 UTC (high liquidity, avoid Asian session noise)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.25 to minimize fee churn

Why this should work:
- Fisher Transform has proven edge in bear markets (Ehlers research)
- ADX regime filter avoids trend-following whipsaws in 2022-2023 range
- 4h/1d HTF ensures we trade with higher TF momentum
- 1h entries give better timing than 4h entries (lower slippage)
- Session filter reduces noise and improves win rate
- Target: 50-80 trades/year on 1h (within fee drag limits)

Entry conditions (LOOSE to guarantee trades):
- LONG trend: ADX>25 + price>4h_HMA>1d_HMA + Fisher crosses above -1.5
- LONG range: ADX<20 + Fisher<-1.8 + price>1d_HMA*0.95
- SHORT trend: ADX>25 + price<4h_HMA<1d_HMA + Fisher crosses below +1.5
- SHORT range: ADX<20 + Fisher>+1.8 + price<1d_HMA*1.05

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_adx_regime_4h1d_session_v1"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    smooth_plus_dm = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    smooth_minus_dm = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    smooth_tr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if smooth_tr[i] > 1e-10:
            plus_di[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
            minus_di[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_fisher(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Extreme values (-2 to +2) indicate reversal points
    More responsive than RSI in choppy/bear markets
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        highest_hl2 = np.max(high[i-period+1:i+1] + low[i-period+1:i+1]) / 2.0
        lowest_hl2 = np.min(high[i-period+1:i+1] + low[i-period+1:i+1]) / 2.0
        
        price_range = highest_hl2 - lowest_hl2
        if price_range < 1e-10:
            continue
        
        normalized = 0.66 * ((hl2 - lowest_hl2) / price_range - 0.5)
        normalized = np.clip(normalized, -0.99, 0.99)
        
        if i > period and not np.isnan(fisher_prev[i-1]):
            fisher_prev[i] = 0.66 * normalized + 0.34 * fisher_prev[i-1]
        else:
            fisher_prev[i] = normalized
        
        if abs(1.0 - fisher_prev[i]) < 1e-10:
            continue
        
        fisher[i] = 0.5 * np.log((1.0 + fisher_prev[i]) / (1.0 - fisher_prev[i]))
    
    return fisher, fisher_prev

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

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
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
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
        
        # === SESSION FILTER (08-20 UTC) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === REGIME DETECTION (ADX) ===
        is_trending = adx_14[i] > 25.0
        is_ranging = adx_14[i] < 20.0
        
        # === HTF BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_4h_bull and hma_1d_bull and hma_4h_aligned[i] > hma_1d_aligned[i]
        strong_bear = hma_4h_bear and hma_1d_bear and hma_4h_aligned[i] < hma_1d_aligned[i]
        
        # === FISHER CROSSOVER DETECTION ===
        fisher_cross_up = fisher_prev[i-1] < -1.5 and fisher[i] > -1.5
        fisher_cross_down = fisher_prev[i-1] > 1.5 and fisher[i] < 1.5
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if in_session:
            if is_trending:
                # TREND FOLLOWING MODE
                if strong_bull and fisher_cross_up and rsi_14[i] > 45:
                    desired_signal = SIZE_STRONG
                elif strong_bear and fisher_cross_down and rsi_14[i] < 55:
                    desired_signal = -SIZE_STRONG
                elif hma_4h_bull and hma_1d_bull and fisher[i] > -1.0:
                    desired_signal = SIZE_BASE
                elif hma_4h_bear and hma_1d_bear and fisher[i] < 1.0:
                    desired_signal = -SIZE_BASE
            
            elif is_ranging:
                # MEAN REVERSION MODE
                if fisher_extreme_low and hma_1d_bull:
                    desired_signal = SIZE_BASE
                elif fisher_extreme_high and hma_1d_bear:
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