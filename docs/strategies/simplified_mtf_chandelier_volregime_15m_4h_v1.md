# Strategy: simplified_mtf_chandelier_volregime_15m_4h_v1

## Status
ACTIVE - Sharpe=5.875 | Return=+36073.9% | DD=-5.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.322 | +1035.2% | -5.6% | 7403 |
| ETHUSDT | 5.738 | +4121.5% | -6.1% | 7370 |
| SOLUSDT | 7.565 | +103064.9% | -5.8% | 7203 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.893 | +46.1% | -5.3% | 2155 |
| ETHUSDT | 5.791 | +189.7% | -5.8% | 2135 |
| SOLUSDT | 7.299 | +354.6% | -4.4% | 2036 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #105 - SIMPLIFIED MTF CHANDELIER WITH VOL REGIME
==================================================================================================
Hypothesis: After #104 crashed due to variable scope issues, this version simplifies:
1. Clean Chandelier exit (3*ATR(22) from highest_high/lowest_low)
2. Volatility regime via BBW percentile for position sizing
3. 15m entries with 4h trend filter (proven 2x Sharpe)
4. Discrete signal levels (0.0, ±0.25, ±0.35) to reduce churn costs
5. Proper variable scoping - all constants defined at function top

Timeframe: 15m entries with 4h trend filter
Key fixes from #104:
- All constants defined before loop
- Cleaner state management
- Proper NaN handling throughout
"""

import numpy as np
import pandas as pd

name = "simplified_mtf_chandelier_volregime_15m_4h_v1"
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


def calculate_hma(close, period=16):
    """Hull Moving Average"""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, w):
        weights = np.arange(1, w + 1)
        return series.rolling(window=w, min_periods=w).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
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


def calculate_adx(high, low, close, period=14):
    """ADX for trend strength"""
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
    
    # ===== Constants (defined before loop to avoid scope issues) =====
    CHANDELIER_MULT = 3.0
    SIZE_LOW_VOL = 0.35
    SIZE_HIGH_VOL = 0.20
    VOL_THRESHOLD = 0.5
    ATR_TARGET_PCT = 0.012
    ADX_MIN = 20
    VOTE_THRESHOLD = 4.0
    VOTE_STREAK_MIN = 2
    FIRST_VALID = 350
    
    # ===== 15m indicators =====
    atr_15m_14 = calculate_atr(high, low, close, period=14)
    atr_15m_22 = calculate_atr(high, low, close, period=22)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    _, st_dir_15m, _ = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi_15m = calculate_rsi(close, period=14)
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
    hma_4h_16 = calculate_hma(c_4h, period=16)
    hma_4h_48 = calculate_hma(c_4h, period=48)
    _, st_dir_4h, atr_4h_22 = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
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
            trend_4h[i] = 1 if c_4h[idx_4h] > hma_4h_16[idx_4h] else (-1 if c_4h[idx_4h] < hma_4h_16[idx_4h] else 0)
            adx_4h_map[i] = adx_4h[idx_4h]
            st_dir_4h_map[i] = st_dir_4h[idx_4h]
            bbw_4h_map[i] = bbw_4h[idx_4h]
            atr_4h_map[i] = atr_4h_22[idx_4h]
    
    # ===== BBW percentile for vol regime =====
    bbw_percentile = np.zeros(n)
    valid_bbw = bbw_4h_map[320:][bbw_4h_map[320:] > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(320, n):
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
        adx_val = adx_4h_map[i]
        st_dir_4h_val = st_dir_4h_map[i]
        bbw_pct = bbw_percentile[i]
        
        # 15m signals
        hma_trend = 1 if hma_16[i] > hma_48[i] else -1
        st_trend = st_dir_15m[i]
        rsi_val = rsi_15m[i]
        
        # Vol regime
        is_low_vol = bbw_pct < VOL_THRESHOLD
        
        # ===== Ensemble voting =====
        vote_long = 0.0
        vote_short = 0.0
        
        # 4h HMA trend (weight: 2.0)
        if trend_4h_val == 1:
            vote_long += 2.0
        elif trend_4h_val == -1:
            vote_short += 2.0
        
        # 4h Supertrend (weight: 1.5)
        if st_dir_4h_val == 1:
            vote_long += 1.5
        elif st_dir_4h_val == -1:
            vote_short += 1.5
        
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
        
        # RSI with SMA200 filter (weight: 0.5)
        if rsi_val > 55 and close[i] > sma_200[i]:
            vote_long += 0.5
        elif rsi_val < 45 and close[i] < sma_200[i]:
            vote_short += 0.5
        
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
        elif vote_streak >= VOTE_STREAK_MIN and adx_val >= ADX_MIN:
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
2026-03-21 11:02
