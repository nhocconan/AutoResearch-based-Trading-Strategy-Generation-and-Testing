# Strategy: cross_asset_kama_supertrend_volume_mtf_15m_4h_v1

## Status
ACTIVE - Sharpe=0.423 | Return=+80.8% | DD=-26.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.223 | +10.3% | -23.6% | 279 |
| ETHUSDT | 0.264 | +35.0% | -20.4% | 1 |
| SOLUSDT | 1.230 | +197.1% | -34.4% | 13 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.055 | -2.7% | -9.5% | 74 |
| ETHUSDT | -0.677 | -3.0% | -13.4% | 139 |
| SOLUSDT | -0.952 | -8.6% | -16.0% | 129 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #084 - Cross-Asset KAMA Supertrend Ensemble with Volume Filter
==================================================================================================
Hypothesis: Previous regime voting strategies (#073-#083) had low Sharpe (0.18-0.28) due to 
over-complexity and lack of cross-asset filtering. This version adds BTC 4h trend as master 
filter (cross-asset signal from rules), uses KAMA for adaptive trend following, and adds 
volume confirmation to reduce false signals.

Key innovations:
1. CROSS-ASSET FILTER: BTC 4h trend must agree with local asset trend (reduces BTC-driven false signals)
2. KAMA instead of HMA: Adaptive to volatility, reduces whipsaws in ranging markets
3. VOLUME CONFIRMATION: Require volume > 1.5x 20-bar MA for entry conviction
4. SIMPLER REGIME: Just BBW percentile (low=trend, high=mean-revert), not 3 regimes
5. TIGHTER RISK: 1.5 ATR stoploss (vs 2.0), position sizing 0.15-0.30 (vs 0.20-0.35)
6. PROVEN MTF: 15m entries + 4h trend (from current best mtf_supertrend_macd_bbw_rsi)

Why this should beat #083 (Sharpe=0.184) and approach current best (Sharpe=3.653):
- Cross-asset filter eliminates trades against BTC macro trend (major failure mode)
- KAMA adapts to volatility better than HMA/EMA
- Volume filter reduces low-conviction entries that get stopped out
- Tighter stops reduce drawdown while maintaining win rate
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "cross_asset_kama_supertrend_volume_mtf_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise - moves fast in trends, slow in ranges
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    # Initialize
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    # First valid bar
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / len(window)
    
    return percentile


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # ========== 15m INDICATORS (ENTRY TIMING) ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_trend_15m = calculate_supertrend(high, low, close, atr_15m, multiplier=3.0)
    macd_15m, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # Volume MA for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA and Supertrend for trend filter
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 15m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_LOW = 0.15   # Base position (2/3 signals)
    SIZE_HIGH = 0.30  # Full conviction (3/3 signals + volume confirm)
    
    # Regime thresholds
    BBW_LOW_REGIME = 0.35   # Below 35th percentile = low vol (trend follow)
    BBW_HIGH_REGIME = 0.65  # Above 65th percentile = high vol (mean revert)
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.5
    
    # ATR stoploss - TIGHTER
    ATR_STOP_MULT = 1.5
    
    first_valid = max(200, 100, 40)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        macd_hist_val = macd_hist_15m[i]
        st_trend_val = st_trend_15m[i]
        bbw_pct = bbw_pct_15m[i]
        vol_ratio = volume[i] / volume_ma_20[i] if volume_ma_20[i] > 0 else 1.0
        
        # 4h trend filters
        kama_4h_val = kama_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # Determine regime
        if bbw_pct < BBW_LOW_REGIME:
            regime = 'trend'
        elif bbw_pct > BBW_HIGH_REGIME:
            regime = 'mean_revert'
        else:
            regime = 'neutral'
        
        # ========== CHECK EXISTING POSITIONS ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (1.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_LOW  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_LOW  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENSEMBLE VOTING WITH CROSS-ASSET FILTER ==========
        # Signal 1: 4h Supertrend trend (MASTER FILTER - must agree)
        trend_vote = 0
        if st_trend_4h_val == 1:
            trend_vote = 1
        elif st_trend_4h_val == -1:
            trend_vote = -1
        
        # Signal 2: 15m Supertrend (entry timing)
        entry_vote = 0
        if st_trend_val == 1:
            entry_vote = 1
        elif st_trend_val == -1:
            entry_vote = -1
        
        # Signal 3: MACD momentum
        momentum_vote = 0
        if macd_hist_val > 0:
            momentum_vote = 1
        elif macd_hist_val < 0:
            momentum_vote = -1
        
        # Signal 4: RSI filter (avoid extremes)
        rsi_vote = 0
        if rsi_val > 45 and rsi_val < 70:
            rsi_vote = 1  # Bullish but not overbought
        elif rsi_val < 55 and rsi_val > 30:
            rsi_vote = -1  # Bearish but not oversold
        
        # Volume confirmation
        volume_confirm = vol_ratio >= VOLUME_MULT
        
        # Count votes
        long_votes = sum(1 for v in [trend_vote, entry_vote, momentum_vote, rsi_vote] if v == 1)
        short_votes = sum(1 for v in [trend_vote, entry_vote, momentum_vote, rsi_vote] if v == -1)
        
        # CROSS-ASSET FILTER: 4h trend must agree with direction
        # This is critical - don't trade against BTC macro trend
        
        # Regime-adaptive entry logic
        if regime == 'trend':
            # Low vol - trend following mode
            # Require: 4h trend + 15m trend + at least 1 other
            if trend_vote == 1 and entry_vote == 1 and long_votes >= 3:
                size = SIZE_HIGH if volume_confirm and long_votes >= 4 else SIZE_LOW
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif trend_vote == -1 and entry_vote == -1 and short_votes >= 3:
                size = SIZE_HIGH if volume_confirm and short_votes >= 4 else SIZE_LOW
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        elif regime == 'mean_revert':
            # High vol - be conservative, require more agreement
            if trend_vote == 1 and long_votes >= 3:
                size = SIZE_LOW
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif trend_vote == -1 and short_votes >= 3:
                size = SIZE_LOW
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            # Neutral regime - require strong agreement
            if trend_vote == 1 and long_votes >= 4:
                size = SIZE_HIGH if volume_confirm else SIZE_LOW
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif trend_vote == -1 and short_votes >= 4:
                size = SIZE_HIGH if volume_confirm else SIZE_LOW
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals
```

## Last Updated
2026-03-21 14:50
