#!/usr/bin/env python3
"""
Experiment #1259: 4h Primary + 1d HTF — Simplified Regime + Volatility Filter

Hypothesis: Recent failures (#1254 Sharpe=-0.186) have correct structure but wrong thresholds.
This strategy simplifies entry logic to ensure >=10 trades/symbol while maintaining edge:
1. CHOPPINESS regime (55/40 thresholds - wider buffer)
2. CONNORS RSI for mean-reversion entries (25/75 - looser than 20/80)
3. 1d HMA(50) for macro trend bias
4. VOLATILITY FILTER: Only enter when ATR > 0.5 * ATR_50 (avoid dead markets)
5. Z-score confirmation for mean-reversion trades

Key changes from #1254:
- Wider chop buffer (55/40 vs 50/45) to reduce regime whipsaw
- Looser CRSI thresholds (25/75 vs stricter values)
- Add volatility filter to avoid low-vol traps
- Simplified position tracking (signal-driven, not state-driven)
- 2.5 ATR trailing stop enforced via signal→0

Target: Sharpe > 0.612, trades >= 30 train, >= 9 test (3 per symbol)
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_crsi_vol_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    hma = np.full(n, np.nan)
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=float)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=float)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection"""
    n = len(close)
    chop = np.full(n, np.nan)
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        if hh > ll and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI"""
    n = len(close)
    crsi = np.full(n, np.nan)
    if n < rank_period + 1:
        return crsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    mask = loss_smooth > 1e-10
    rsi_short[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_short[:rsi_period] = np.nan
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[:streak_period] = np.nan
    
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(window < current)
        pct_rank[i] = 100.0 * rank / rank_period
    pct_rank[:rank_period] = np.nan
    
    valid = (~np.isnan(rsi_short)) & (~np.isnan(rsi_streak)) & (~np.isnan(pct_rank))
    crsi[valid] = (rsi_short[valid] + rsi_streak[valid] + pct_rank[valid]) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_long(close, high, low, period=50):
    """Long-period ATR for volatility filter"""
    n = len(close)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return mid, upper, lower
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        mid[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    return mid, upper, lower

def calculate_zscore(close, period=20):
    """Z-score for mean reversion"""
    n = len(close)
    zscore = np.full(n, np.nan)
    if n < period:
        return zscore
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        mean = np.mean(window)
        std = np.std(window, ddof=0)
        if std > 1e-10:
            zscore[i] = (close[i] - mean) / std
    return zscore

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr_long(close, high, low, period=50)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    zscore = calculate_zscore(close, period=20)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Track entry for stoploss
    entry_price = np.zeros(n)
    entry_atr = np.zeros(n)
    highest_since = np.zeros(n)
    lowest_since = np.full(n, np.inf)
    position_side = np.zeros(n)
    
    for i in range(150, n):
        # Validate indicators
        if np.isnan(atr[i]) or np.isnan(atr_50[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(bb_mid[i]):
            signals[i] = 0.0
            continue
        if np.isnan(zscore[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY FILTER ===
        vol_ok = atr[i] > 0.5 * atr_50[i]
        if not vol_ok:
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        in_range = chop[i] > 55.0
        in_trend = chop[i] < 40.0
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (4h HMA) ===
        local_bull = hma_21[i] > hma_50[i]
        local_bear = hma_21[i] < hma_50[i]
        
        # === TREND STRENGTH ===
        trend_strong = adx[i] > 20.0
        
        desired_signal = 0.0
        
        # === RANGING REGIME: Mean Reversion ===
        if in_range:
            # Long: CRSI oversold + price at BB lower + zscore negative
            if crsi[i] < 25.0 and close[i] <= bb_lower[i] * 1.005 and zscore[i] < -1.0:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + price at BB upper + zscore positive
            elif crsi[i] > 75.0 and close[i] >= bb_upper[i] * 0.995 and zscore[i] > 1.0:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME: Trend Following ===
        elif in_trend:
            # Long: Macro bull + Local bull + ADX strong + CRSI not overbought
            if macro_bull and local_bull and trend_strong and crsi[i] < 60.0:
                desired_signal = BASE_SIZE
            # Short: Macro bear + Local bear + ADX strong + CRSI not oversold
            elif macro_bear and local_bear and trend_strong and crsi[i] > 40.0:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL/TRANSITION: Pullback entries ===
        else:
            # Long pullback in uptrend
            if macro_bull and local_bull and crsi[i] < 35.0:
                desired_signal = BASE_SIZE * 0.5
            # Short pullback in downtrend
            elif macro_bear and local_bear and crsi[i] > 65.0:
                desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if i > 0 and position_side[i-1] != 0:
            prev_side = position_side[i-1]
            prev_entry = entry_price[i-1]
            prev_atr = entry_atr[i-1]
            
            if prev_side > 0:
                trail_high = max(prev_entry, highest_since[i-1] if highest_since[i-1] > 0 else prev_entry)
                stop_price = trail_high - 2.5 * prev_atr
                if close[i] < stop_price:
                    desired_signal = 0.0
            elif prev_side < 0:
                trail_low = min(prev_entry, lowest_since[i-1] if lowest_since[i-1] < np.inf else prev_entry)
                stop_price = trail_low + 2.5 * prev_atr
                if close[i] > stop_price:
                    desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        signals[i] = final_signal
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            position_side[i] = int(np.sign(final_signal))
            if i > 0 and position_side[i-1] == 0:
                entry_price[i] = close[i]
                entry_atr[i] = atr[i]
                if position_side[i] > 0:
                    highest_since[i] = close[i]
                    lowest_since[i] = np.inf
                else:
                    lowest_since[i] = close[i]
                    highest_since[i] = 0.0
            elif i > 0 and position_side[i-1] != 0:
                entry_price[i] = entry_price[i-1]
                entry_atr[i] = entry_atr[i-1]
                if position_side[i] > 0:
                    highest_since[i] = max(highest_since[i-1], close[i])
                    lowest_since[i] = lowest_since[i-1]
                else:
                    lowest_since[i] = min(lowest_since[i-1], close[i])
                    highest_since[i] = highest_since[i-1]
        else:
            position_side[i] = 0.0
            entry_price[i] = 0.0
            entry_atr[i] = 0.0
            highest_since[i] = 0.0
            lowest_since[i] = np.inf
    
    return signals