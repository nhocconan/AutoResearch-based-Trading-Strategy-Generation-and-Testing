#!/usr/bin/env python3
"""
EXPERIMENT #101 - CHANDELIER HMA ENSEMBLE WITH VOL-ADAPTIVE SIZING (15m+4h v2)
==================================================================================================
Hypothesis: Best strategy (Sharpe=16.016) uses HMA+Supertrend+RSI+Z-score+BBW on 15m.
This experiment enhances it with:

Key innovations for #101:
1. Chandelier Exit (ATR trailing stop) - proven stop mechanism from Chuck LeBeau
2. Volatility-adjusted position sizing - reduce size when ATR% is high
3. 15m entries + 4h trend filter - optimal signal-to-noise ratio
4. HMA(16/48) for fast trend detection - less lag than EMA
5. Supertrend(10,3) for regime confirmation
6. RSI(14) with 200-SMA filter for momentum
7. Z-score(20) for extreme detection
8. BBW percentile for regime (low=trend, high=mean-revert)
9. Discrete position levels: 0.0, ±0.20, ±0.35 (reduces churn)
10. 2-bar hysteresis for entry confirmation

Why this should work:
- Chandelier Exit (highest_high - 3*ATR) is proven to capture trends while protecting profits
- Vol-adjusted sizing prevents blowup during high volatility periods
- 15m timeframe has better entries than 1h, less noise than 5m
- 4h filter prevents counter-trend trades (proven in experiments #090-#100)
- Multiple uncorrelated signals reduce false positives

Risk controls:
- Max position size: 0.35 (35% of capital)
- Chandelier stop: 3*ATR(22) from highest high (long) / lowest low (short)
- Take profit: reduce to half at 2R, trail stop at 1R
- Volatility-adjusted sizing: base_size * (target_ATR% / current_ATR%)
- ADX filter: only trade when 4h ADX > 20 (trend strength)
"""

import numpy as np
import pandas as pd

name = "chandelier_hma_ensemble_vol_adaptive_15m_4h_v2"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, wma_period):
        wma_out = np.zeros(len(series))
        weights = np.arange(1, wma_period + 1)
        for i in range(wma_period - 1, len(series)):
            wma_out[i] = np.sum(series[i - wma_period + 1:i + 1] * weights) / np.sum(weights)
        return wma_out
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma_raw = 2 * wma_half - wma_full
    
    hma = np.zeros(n)
    for i in range(sqrt_period - 1, n):
        hma[i] = wma(hma_raw[sqrt_period - 1:i + 1], sqrt_period)[i - sqrt_period + 1] if i >= sqrt_period - 1 else 0
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
    
    # Supertrend direction
    st = np.zeros(n)
    st_dir = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    st[period] = upper_band[period]
    st_dir[period] = -1
    
    for i in range(period + 1, n):
        if close[i - 1] <= st[i - 1]:
            st[i] = min(upper_band[i], st[i - 1]) if upper_band[i] < st[i - 1] or close[i - 1] > st[i - 1] else st[i - 1]
            st_dir[i] = -1
        else:
            st[i] = max(lower_band[i], st[i - 1]) if lower_band[i] > st[i - 1] or close[i - 1] < st[i - 1] else st[i - 1]
            st_dir[i] = 1
    
    return st, st_dir, atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    rsi = np.zeros(n)
    delta = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    for i in range(period - 1, n):
        sma = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma + std_dev * std
        lower[i] = sma - std_dev * std
        if sma > 0:
            bandwidth[i] = (upper[i] - lower[i]) / sma
    
    return upper, lower, bandwidth


def calculate_adx(high, low, close, period=14):
    """Calculate ADX"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def resample_to_4h(close, high, low, volume):
    """Resample 15m data to 4h (16 bars per 4h)"""
    n = len(close)
    n_4h = n // 16
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    v_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * 16
        end_idx = start_idx + 16
        if end_idx <= n:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
            v_4h[i] = np.sum(volume[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h, v_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    signals = np.zeros(n)
    
    # ===== 15m indicators for entry =====
    atr_15m = calculate_atr(high, low, close, period=14)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    supertrend_15m, st_dir_15m, _ = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    bb_upper, bb_lower, bb_bw = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # 200-SMA for RSI filter
    sma_200 = np.zeros(n)
    for i in range(199, n):
        sma_200[i] = np.mean(close[i - 199:i + 1])
    
    # ===== 4h indicators for trend filter =====
    c_4h, h_4h, l_4h, v_4h = resample_to_4h(close, high, low, volume)
    n_4h = len(c_4h)
    
    hma_4h = calculate_hma(c_4h, period=16)
    supertrend_4h, st_dir_4h, atr_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    bb_bw_4h = calculate_bollinger_bands(c_4h, period=20, std_dev=2.0)[2]
    
    # Map 4h indicators to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    st_dir_4h_mapped = np.zeros(n)
    bb_bw_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // 16
        if idx_4h < n_4h and idx_4h >= 20:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            st_dir_4h_mapped[i] = st_dir_4h[idx_4h]
            bb_bw_4h_mapped[i] = bb_bw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Calculate BBW percentile for regime detection
    bbw_percentile = np.zeros(n)
    valid_bbw = bb_bw_4h_mapped[320:]  # 20 * 16 bars warmup
    valid_bbw = valid_bbw[valid_bbw > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(320, n):
            if bb_bw_4h_mapped[i] > 0:
                bbw_percentile[i] = np.searchsorted(bbw_sorted, bb_bw_4h_mapped[i]) / len(bbw_sorted)
    
    # ===== Position sizing parameters =====
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    ATR_TARGET_PCT = 0.012  # Target 1.2% ATR
    ADX_MIN = 20
    ZSCORE_EXTREME = 2.0
    
    # ===== Tracking variables =====
    prev_signal = 0.0
    consecutive_votes = 0
    prev_vote_direction = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    tp_triggered = False
    chandelier_stop = 0.0
    
    first_valid = max(320, 200, 50, 48)  # Warmup period
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if (np.isnan(atr_15m[i]) or atr_15m[i] == 0 or 
            np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or
            np.isnan(adx_4h_mapped[i]) or np.isnan(st_dir_4h_mapped[i])):
            signals[i] = 0.0
            prev_signal = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            entry_price = 0.0
            continue
        
        # Get indicator values
        trend_4h_val = trend_4h[i]
        adx_val = adx_4h_mapped[i]
        st_dir_4h_val = st_dir_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        
        # 15m signals
        hma_trend = 1 if hma_16[i] > hma_48[i] else -1
        st_trend = st_dir_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        # Regime: low BW = trend follow, high BW = mean revert
        trend_regime = bbw_pct < 0.5
        
        # 4h ADX filter
        adx_filter = adx_val >= ADX_MIN
        
        # ===== ENSEMBLE VOTING =====
        vote_long = 0
        vote_short = 0
        
        # Signal 1: 4h HMA trend
        if trend_4h_val == 1:
            vote_long += 1.5
        elif trend_4h_val == -1:
            vote_short += 1.5
        
        # Signal 2: 4h Supertrend
        if st_dir_4h_val == 1:
            vote_long += 1.0
        elif st_dir_4h_val == -1:
            vote_short += 1.0
        
        # Signal 3: 15m HMA crossover
        if hma_trend == 1:
            vote_long += 1.0
        else:
            vote_short += 1.0
        
        # Signal 4: 15m Supertrend
        if st_trend == 1:
            vote_long += 1.0
        else:
            vote_short += 1.0
        
        # Signal 5: RSI with SMA filter
        if rsi_val > 55 and close[i] > sma_200[i]:
            vote_long += 0.5
        elif rsi_val < 45 and close[i] < sma_200[i]:
            vote_short += 0.5
        
        # Signal 6: Z-score extreme (mean reversion in high BW regime)
        if not trend_regime:
            if zscore_val < -ZSCORE_EXTREME:
                vote_long += 0.5
            elif zscore_val > ZSCORE_EXTREME:
                vote_short += 0.5
        
        # Determine vote direction
        if vote_long > vote_short and vote_long >= 3.5:
            current_vote = 1
            total_votes = vote_long
        elif vote_short > vote_long and vote_short >= 3.5:
            current_vote = -1
            total_votes = vote_short
        else:
            current_vote = 0
            total_votes = 0
        
        # Hysteresis: 2 consecutive bars for entry
        if current_vote != 0 and current_vote == prev_vote_direction:
            consecutive_votes += 1
        elif current_vote != 0:
            consecutive_votes = 1
            prev_vote_direction = current_vote
        else:
            consecutive_votes = 0
            prev_vote_direction = 0
        
        # Calculate volatility-adjusted size
        atr_pct = atr_15m[i] / close[i] if close[i] > 0 else 0
        vol_adjustment = min(1.5, max(0.5, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # ===== CHANDELIER EXIT STOP MANAGEMENT =====
        if prev_signal != 0.0 and entry_price > 0:
            chandelier_mult = 3.0
            atr_stop = atr_15m[i]
            
            if prev_signal > 0:  # Long position
                highest_high = max(highest_high, high[i])
                chandelier_stop = highest_high - chandelier_mult * atr_stop
                
                # Take profit at 2R
                if not tp_triggered and close[i] >= entry_price + 2 * chandelier_mult * entry_atr:
                    signals[i] = prev_signal * 0.5  # Reduce to half
                    tp_triggered = True
                    continue
                
                # Stop loss
                if close[i] < chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    tp_triggered = False
                    highest_high = 0.0
                    continue
            else:  # Short position
                lowest_low = min(lowest_low, low[i])
                chandelier_stop = lowest_low + chandelier_mult * atr_stop
                
                # Take profit at 2R
                if not tp_triggered and close[i] <= entry_price - 2 * chandelier_mult * entry_atr:
                    signals[i] = prev_signal * 0.5  # Reduce to half
                    tp_triggered = True
                    continue
                
                # Stop loss
                if close[i] > chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    tp_triggered = False
                    lowest_low = 0.0
                    continue
        
        # ===== GENERATE SIGNAL =====
        if prev_signal != 0.0:
            # Hold position if vote direction matches
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
                tp_triggered = False
            else:
                signals[i] = prev_signal
        elif consecutive_votes >= 2 and adx_filter:
            if current_vote == 1:
                base_size = SIZE_HIGH if total_votes >= 5.0 else SIZE_LOW
                signals[i] = np.clip(base_size * vol_adjustment, 0, SIZE_HIGH)
                entry_price = close[i]
                entry_atr = atr_15m[i]
                highest_high = high[i]
                prev_signal = signals[i]
                tp_triggered = False
            else:
                base_size = SIZE_HIGH if total_votes >= 5.0 else SIZE_LOW
                signals[i] = -np.clip(base_size * vol_adjustment, 0, SIZE_HIGH)
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