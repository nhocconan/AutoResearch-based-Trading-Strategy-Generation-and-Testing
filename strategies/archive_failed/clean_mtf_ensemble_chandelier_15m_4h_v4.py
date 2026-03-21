#!/usr/bin/env python3
"""
EXPERIMENT #103 - CLEAN MTF ENSEMBLE WITH CHANDELIER EXIT (15m+4h v4)
==================================================================================================
Hypothesis: Experiment #102 failed due to read-only array assignment issues. This version fixes
all array mutability problems while keeping the proven ensemble logic that achieved Sharpe=16+.

Key fixes for #103:
1. All indicator arrays created as writable numpy arrays (no views)
2. Simplified 4h resampling with proper array initialization
3. Cleaner state management with fewer variables
4. Maintain proven components: HMA trend, Supertrend, ADX filter, Chandelier exit
5. Volatility-adjusted position sizing with discrete levels (0.20, 0.35)
6. Reduced signal churn with vote streak hysteresis

Risk controls:
- Max position size: 0.35 (35% of capital)
- Chandelier stop: 3*ATR(22) from highest high (long) / lowest low (short)
- Volatility-adjusted sizing: base_size * (target_ATR% / current_ATR%)
- ADX filter: only trade when 4h ADX > 20 (trend strength)
- Take profit: reduce to half at 2R, trail stop
"""

import numpy as np
import pandas as pd

name = "clean_mtf_ensemble_chandelier_15m_4h_v4"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """ATR calculation with proper warmup"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=16):
    """Hull Moving Average"""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, w):
        weights = np.arange(1, w + 1)
        return series.rolling(window=w, min_periods=w).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma.values.copy()


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend with direction"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    supertrend[period] = upper[period]
    direction[period] = -1
    
    for i in range(period + 1, n):
        if close[i-1] <= supertrend[i-1]:
            supertrend[i] = min(upper[i], supertrend[i-1])
            direction[i] = -1
        else:
            supertrend[i] = max(lower[i], supertrend[i-1])
            direction[i] = 1
    
    return supertrend.copy(), direction.copy(), atr.copy()


def calculate_rsi(close, period=14):
    """RSI calculation"""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    
    avg_gain = gain_s.rolling(window=period, min_periods=period).mean()
    avg_loss = loss_s.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values.copy()


def calculate_zscore(close, period=20):
    """Z-score calculation"""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    zscore = (close_s - mean) / std
    return zscore.fillna(0).values.copy()


def calculate_bbw(close, period=20, std_dev=2.0):
    """Bollinger Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return bandwidth.fillna(0).values.copy()


def calculate_adx(high, low, close, period=14):
    """ADX calculation"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    
    for i in range(period, n):
        if di_sum[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum[i]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
    
    return adx.fillna(0).values.copy()


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    signals = np.zeros(n)
    
    # ===== 15m indicators =====
    atr_15m = calculate_atr(high, low, close, period=14)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    _, st_dir_15m, _ = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    bbw_15m = calculate_bbw(close, period=20, std_dev=2.0)
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values.copy()
    sma_200 = np.nan_to_num(sma_200, 0)
    
    # ===== 4h resampling (16 bars per 4h on 15m) =====
    bars_per_4h = 16
    n_4h = n // bars_per_4h
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = min(start_idx + bars_per_4h, n)
        if end_idx > start_idx:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
    
    # ===== 4h indicators =====
    hma_4h = calculate_hma(c_4h, period=16)
    _, st_dir_4h, atr_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    bbw_4h = calculate_bbw(c_4h, period=20, std_dev=2.0)
    
    # ===== Map 4h to 15m =====
    trend_4h = np.zeros(n)
    adx_4h_map = np.zeros(n)
    st_dir_4h_map = np.zeros(n)
    bbw_4h_map = np.zeros(n)
    atr_4h_map = np.zeros(n)
    
    for i in range(n):
        idx_4h = min(i // bars_per_4h, n_4h - 1)
        if idx_4h >= 20:
            trend_4h[i] = 1 if c_4h[idx_4h] > hma_4h[idx_4h] else (-1 if c_4h[idx_4h] < hma_4h[idx_4h] else 0)
            adx_4h_map[i] = adx_4h[idx_4h]
            st_dir_4h_map[i] = st_dir_4h[idx_4h]
            bbw_4h_map[i] = bbw_4h[idx_4h]
            atr_4h_map[i] = atr_4h[idx_4h]
    
    # ===== BBW percentile for regime =====
    bbw_percentile = np.zeros(n)
    valid_bbw = bbw_4h_map[320:][bbw_4h_map[320:] > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(320, n):
            if bbw_4h_map[i] > 0:
                bbw_percentile[i] = np.searchsorted(bbw_sorted, bbw_4h_map[i]) / len(bbw_sorted)
    
    # ===== Parameters =====
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    ATR_TARGET_PCT = 0.012
    ADX_MIN = 20
    ZSCORE_EXTREME = 2.0
    FIRST_VALID = 350
    
    # ===== State =====
    prev_signal = 0.0
    prev_vote = 0
    vote_streak = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    tp_triggered = False
    
    for i in range(FIRST_VALID, n):
        if atr_15m[i] == 0 or np.isnan(atr_15m[i]) or close[i] == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # 4h filters
        trend_4h_val = trend_4h[i]
        adx_val = adx_4h_map[i]
        st_dir_4h_val = st_dir_4h_map[i]
        bbw_pct = bbw_percentile[i]
        
        # 15m signals
        hma_trend = 1 if hma_16[i] > hma_48[i] else -1
        st_trend = st_dir_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        # Regime
        trend_regime = bbw_pct < 0.5
        adx_filter = adx_val >= ADX_MIN
        
        # ===== Ensemble voting =====
        vote_long = 0.0
        vote_short = 0.0
        
        # 4h HMA trend (weight: 1.5)
        if trend_4h_val == 1:
            vote_long += 1.5
        elif trend_4h_val == -1:
            vote_short += 1.5
        
        # 4h Supertrend (weight: 1.0)
        if st_dir_4h_val == 1:
            vote_long += 1.0
        elif st_dir_4h_val == -1:
            vote_short += 1.0
        
        # 15m HMA crossover (weight: 1.0)
        if hma_trend == 1:
            vote_long += 1.0
        else:
            vote_short += 1.0
        
        # 15m Supertrend (weight: 1.0)
        if st_trend == 1:
            vote_long += 1.0
        else:
            vote_short += 1.0
        
        # RSI with SMA filter (weight: 0.5)
        if rsi_val > 55 and close[i] > sma_200[i]:
            vote_long += 0.5
        elif rsi_val < 45 and close[i] < sma_200[i]:
            vote_short += 0.5
        
        # Z-score mean reversion in high BW regime (weight: 0.5)
        if not trend_regime:
            if zscore_val < -ZSCORE_EXTREME:
                vote_long += 0.5
            elif zscore_val > ZSCORE_EXTREME:
                vote_short += 0.5
        
        # Determine vote
        if vote_long > vote_short and vote_long >= 3.5:
            current_vote = 1
            total_votes = vote_long
        elif vote_short > vote_long and vote_short >= 3.5:
            current_vote = -1
            total_votes = vote_short
        else:
            current_vote = 0
            total_votes = 0
        
        # Vote streak
        if current_vote != 0 and current_vote == prev_vote:
            vote_streak += 1
        elif current_vote != 0:
            vote_streak = 1
            prev_vote = current_vote
        else:
            vote_streak = 0
            prev_vote = 0
        
        # Volatility adjustment
        atr_pct = atr_15m[i] / close[i] if close[i] > 0 else 0
        vol_adj = min(1.5, max(0.5, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # ===== Chandelier Exit management =====
        if prev_signal != 0.0 and entry_price > 0:
            chandelier_mult = 3.0
            atr_stop = atr_15m[i]
            
            if prev_signal > 0:  # Long
                highest_high = max(highest_high, high[i])
                chandelier_stop = highest_high - chandelier_mult * atr_stop
                
                # Take profit at 2R
                if not tp_triggered and close[i] >= entry_price + 2 * chandelier_mult * entry_atr:
                    signals[i] = prev_signal * 0.5
                    tp_triggered = True
                    continue
                
                # Stop loss
                if close[i] < chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    vote_streak = 0
                    tp_triggered = False
                    highest_high = 0.0
                    continue
            else:  # Short
                lowest_low = min(lowest_low, low[i])
                chandelier_stop = lowest_low + chandelier_mult * atr_stop
                
                # Take profit at 2R
                if not tp_triggered and close[i] <= entry_price - 2 * chandelier_mult * entry_atr:
                    signals[i] = prev_signal * 0.5
                    tp_triggered = True
                    continue
                
                # Stop loss
                if close[i] > chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    vote_streak = 0
                    tp_triggered = False
                    lowest_low = 0.0
                    continue
        
        # ===== Generate signal =====
        if prev_signal != 0.0:
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
                tp_triggered = False
            else:
                signals[i] = prev_signal
        elif vote_streak >= 2 and adx_filter:
            if current_vote == 1:
                base_size = SIZE_HIGH if total_votes >= 5.0 else SIZE_LOW
                signals[i] = np.clip(base_size * vol_adj, 0, SIZE_HIGH)
                entry_price = close[i]
                entry_atr = atr_15m[i]
                highest_high = high[i]
                prev_signal = signals[i]
                tp_triggered = False
            else:
                base_size = SIZE_HIGH if total_votes >= 5.0 else SIZE_LOW
                signals[i] = -np.clip(base_size * vol_adj, 0, SIZE_HIGH)
                entry_price = close[i]
                entry_atr = atr_15m[i]
                lowest_low = low[i]
                prev_signal = signals[i]
                tp_triggered = False
        else:
            signals[i] = 0.0
            prev_signal = 0.0
    
    # Clip to max position size
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals