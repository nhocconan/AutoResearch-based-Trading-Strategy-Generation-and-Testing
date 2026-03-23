# Strategy: kama_donchian_zscore_mtf_chandelier_15m_4h_v2

## Status
ACTIVE - Sharpe=1.509 | Return=+842.1% | DD=-24.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.721 | -11.2% | -38.2% | 11007 |
| ETHUSDT | 1.071 | +103.3% | -26.5% | 11307 |
| SOLUSDT | 4.178 | +2434.2% | -8.9% | 11360 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.841 | -10.7% | -19.4% | 3149 |
| ETHUSDT | 2.909 | +59.6% | -7.2% | 3133 |
| SOLUSDT | 3.654 | +92.2% | -13.5% | 3187 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #106 - KAMA DONCHIAN ZSCORE MTF CHANDELIER V2
==================================================================================================
Hypothesis: Combine adaptive trend (KAMA) with breakout detection (Donchian) and mean reversion
filter (Z-score) across 15m/4h timeframes. KAMA adapts to volatility better than HMA/EMA.
Donchian breakouts capture momentum moves. Z-score prevents chasing overextended prices.
Chandelier exit provides dynamic trailing stop. BBW percentile adjusts position size.

Timeframe: 15m entries with 4h trend filter
Key features:
1. KAMA(10,2,30) for adaptive trend following
2. Donchian(20) breakout detection
3. Z-score(20) for overextension filter
4. 4h KAMA trend filter
5. ATR(22) Chandelier exit (3*ATR)
6. BBW percentile for vol regime sizing
7. Discrete signal levels (0.0, ±0.25, ±0.35)
"""

import numpy as np
import pandas as pd

name = "kama_donchian_zscore_mtf_chandelier_15m_4h_v2"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """ATR with Wilder's smoothing"""
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(er_period))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothing constant
    sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bounds"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower


def calculate_zscore(close, period=20):
    """Z-score for mean reversion detection"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.fillna(0).values.copy()


def calculate_rsi(close, period=14):
    """RSI with proper smoothing"""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    
    avg_gain = gain_s.ewm(span=period, min_periods=period).mean()
    avg_loss = loss_s.ewm(span=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values.copy()


def calculate_bbw(close, period=20, std_dev=2.0):
    """Bollinger Band Width for volatility regime"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return bandwidth.fillna(0).values.copy()


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    signals = np.zeros(n)
    
    # ===== Constants =====
    CHANDELIER_MULT = 3.0
    SIZE_LOW_VOL = 0.35
    SIZE_HIGH_VOL = 0.20
    VOL_THRESHOLD = 0.5
    ATR_TARGET_PCT = 0.012
    ZSCORE_MAX = 2.0
    RSI_LONG_MIN = 45
    RSI_SHORT_MAX = 55
    VOTE_THRESHOLD = 3.5
    VOTE_STREAK_MIN = 2
    FIRST_VALID = 400
    
    # ===== 15m indicators =====
    atr_15m_14 = calculate_atr(high, low, close, period=14)
    atr_15m_22 = calculate_atr(high, low, close, period=22)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    zscore_15m = calculate_zscore(close, period=20)
    rsi_15m = calculate_rsi(close, period=14)
    bbw_15m = calculate_bbw(close, period=20, std_dev=2.0)
    
    # KAMA slope
    kama_slope_15m = np.zeros(n)
    for i in range(1, n):
        if kama_15m[i-1] != 0:
            kama_slope_15m[i] = (kama_15m[i] - kama_15m[i-1]) / kama_15m[i-1]
    
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
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
    bbw_4h = calculate_bbw(c_4h, period=20, std_dev=2.0)
    atr_4h_22 = calculate_atr(h_4h, l_4h, c_4h, period=22)
    
    # KAMA slope 4h
    kama_slope_4h = np.zeros(n_4h)
    for i in range(1, n_4h):
        if kama_4h[i-1] != 0:
            kama_slope_4h[i] = (kama_4h[i] - kama_4h[i-1]) / kama_4h[i-1]
    
    # ===== Map 4h to 15m =====
    trend_4h = np.zeros(n)
    bbw_4h_map = np.zeros(n)
    atr_4h_map = np.zeros(n)
    kama_slope_4h_map = np.zeros(n)
    donchian_breakout_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = min(i // bars_per_4h, n_4h - 1)
        if idx_4h >= 20:
            # Trend direction
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                trend_4h[i] = -1
            
            bbw_4h_map[i] = bbw_4h[idx_4h]
            atr_4h_map[i] = atr_4h_22[idx_4h]
            kama_slope_4h_map[i] = kama_slope_4h[idx_4h]
            
            # Donchian breakout detection
            if c_4h[idx_4h] >= donchian_upper_4h[idx_4h] * 0.999:
                donchian_breakout_4h[i] = 1
            elif c_4h[idx_4h] <= donchian_lower_4h[idx_4h] * 1.001:
                donchian_breakout_4h[i] = -1
    
    # ===== BBW percentile for vol regime =====
    bbw_percentile = np.zeros(n)
    valid_bbw = bbw_4h_map[FIRST_VALID:][bbw_4h_map[FIRST_VALID:] > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(FIRST_VALID, n):
            if bbw_4h_map[i] > 0:
                bbw_percentile[i] = np.searchsorted(bbw_sorted, bbw_4h_map[i]) / len(bbw_sorted)
    
    # ===== State variables =====
    prev_signal = 0.0
    prev_vote = 0
    vote_streak = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    tp_triggered = False
    chandelier_stop = 0.0
    
    for i in range(FIRST_VALID, n):
        # Skip invalid data
        if atr_15m_22[i] == 0 or np.isnan(atr_15m_22[i]) or close[i] == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # 4h filters
        trend_4h_val = trend_4h[i]
        bbw_pct = bbw_percentile[i]
        breakout_4h = donchian_breakout_4h[i]
        kama_slope_4h_val = kama_slope_4h_map[i]
        
        # 15m signals
        kama_trend_15m = 1 if close[i] > kama_15m[i] else -1
        kama_slope_15m_val = kama_slope_15m[i]
        zscore_val = zscore_15m[i]
        rsi_val = rsi_15m[i]
        
        # Donchian breakout 15m
        donchian_breakout_15m = 0
        if close[i] >= donchian_upper_15m[i] * 0.999:
            donchian_breakout_15m = 1
        elif close[i] <= donchian_lower_15m[i] * 1.001:
            donchian_breakout_15m = -1
        
        # Vol regime
        is_low_vol = bbw_pct < VOL_THRESHOLD
        
        # ===== Ensemble voting =====
        vote_long = 0.0
        vote_short = 0.0
        
        # 4h KAMA trend (weight: 2.0)
        if trend_4h_val == 1:
            vote_long += 2.0
        elif trend_4h_val == -1:
            vote_short += 2.0
        
        # 4h KAMA slope confirmation (weight: 1.0)
        if kama_slope_4h_val > 0.001:
            vote_long += 1.0
        elif kama_slope_4h_val < -0.001:
            vote_short += 1.0
        
        # 4h Donchian breakout (weight: 1.5)
        if breakout_4h == 1:
            vote_long += 1.5
        elif breakout_4h == -1:
            vote_short += 1.5
        
        # 15m KAMA trend (weight: 1.0)
        if kama_trend_15m == 1:
            vote_long += 1.0
        else:
            vote_short += 1.0
        
        # 15m Donchian breakout (weight: 1.0)
        if donchian_breakout_15m == 1:
            vote_long += 1.0
        elif donchian_breakout_15m == -1:
            vote_short += 1.0
        
        # RSI filter (weight: 0.5)
        if rsi_val > RSI_LONG_MIN and rsi_val < 70:
            vote_long += 0.5
        elif rsi_val < RSI_SHORT_MAX and rsi_val > 30:
            vote_short += 0.5
        
        # Z-score overextension filter (penalty)
        if zscore_val > ZSCORE_MAX:
            vote_long -= 2.0
        elif zscore_val < -ZSCORE_MAX:
            vote_short -= 2.0
        
        # Determine vote
        if vote_long > vote_short and vote_long >= VOTE_THRESHOLD:
            current_vote = 1
        elif vote_short > vote_long and vote_short >= VOTE_THRESHOLD:
            current_vote = -1
        else:
            current_vote = 0
        
        # Vote streak for hysteresis
        if current_vote != 0 and current_vote == prev_vote:
            vote_streak += 1
        elif current_vote != 0:
            vote_streak = 1
            prev_vote = current_vote
        else:
            vote_streak = 0
            prev_vote = 0
        
        # Volatility-adjusted position sizing
        atr_pct = atr_15m_22[i] / close[i] if close[i] > 0 else 0
        vol_adj = min(1.3, max(0.7, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # ===== Chandelier Exit management =====
        if prev_signal != 0.0 and entry_price > 0:
            atr_stop = atr_15m_22[i]
            
            if prev_signal > 0:  # Long position
                highest_high = max(highest_high, high[i])
                chandelier_stop = highest_high - CHANDELIER_MULT * atr_stop
                
                # Take profit at 2R - reduce to half
                if not tp_triggered and close[i] >= entry_price + 2 * CHANDELIER_MULT * entry_atr:
                    signals[i] = prev_signal * 0.5
                    tp_triggered = True
                    chandelier_stop = max(chandelier_stop, entry_price + CHANDELIER_MULT * entry_atr)
                    prev_signal = signals[i]
                    continue
                
                # Stop loss - Chandelier exit
                if close[i] < chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    vote_streak = 0
                    tp_triggered = False
                    highest_high = 0.0
                    continue
                    
            else:  # Short position
                lowest_low = min(lowest_low, low[i])
                chandelier_stop = lowest_low + CHANDELIER_MULT * atr_stop
                
                # Take profit at 2R - reduce to half
                if not tp_triggered and close[i] <= entry_price - 2 * CHANDELIER_MULT * entry_atr:
                    signals[i] = prev_signal * 0.5
                    tp_triggered = True
                    chandelier_stop = min(chandelier_stop, entry_price - CHANDELIER_MULT * entry_atr)
                    prev_signal = signals[i]
                    continue
                
                # Stop loss - Chandelier exit
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
            # Hold position if vote agrees, exit if vote flips
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
                tp_triggered = False
            else:
                signals[i] = prev_signal
        elif vote_streak >= VOTE_STREAK_MIN:
            # New entry with volatility-adjusted sizing
            base_size = SIZE_LOW_VOL if is_low_vol else SIZE_HIGH_VOL
            position_size = np.clip(base_size * vol_adj, 0.15, SIZE_LOW_VOL)
            
            if current_vote == 1:
                signals[i] = position_size
                entry_price = close[i]
                entry_atr = atr_15m_22[i]
                highest_high = high[i]
                chandelier_stop = highest_high - CHANDELIER_MULT * entry_atr
                prev_signal = signals[i]
                tp_triggered = False
            else:
                signals[i] = -position_size
                entry_price = close[i]
                entry_atr = atr_15m_22[i]
                lowest_low = low[i]
                chandelier_stop = lowest_low + CHANDELIER_MULT * entry_atr
                prev_signal = signals[i]
                tp_triggered = False
        else:
            signals[i] = 0.0
            prev_signal = 0.0
    
    # Clip to max position size
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals
```

## Last Updated
2026-03-21 11:03
