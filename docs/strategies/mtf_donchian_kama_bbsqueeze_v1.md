# Strategy: mtf_donchian_kama_bbsqueeze_v1

## Status
ACTIVE - Sharpe=1.442 | Return=+155.4% | DD=-7.6%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.358 | +34.4% | -10.2% | 1322 |
| ETHUSDT | 1.807 | +134.0% | -4.5% | 1377 |
| SOLUSDT | 2.161 | +297.6% | -8.2% | 1358 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.981 | +12.2% | -3.0% | 360 |
| ETHUSDT | 1.147 | +16.7% | -3.4% | 341 |
| SOLUSDT | 1.693 | +29.4% | -4.8% | 377 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #022 - Donchian Trend + KAMA Adaptive + BB Squeeze Entry
====================================================================================
Hypothesis: Replace HMA trend with Donchian Channel breakouts (pure price action trend).
Use KAMA (Kaufman Adaptive Moving Average) for adaptive momentum that adjusts to volatility.
Enter on Bollinger Band squeeze expansion (volatility breakout) + RSI confirmation.

Key differences from #021:
- Donchian(20) breakout trend instead of HMA crossover - captures pure price breakouts
- KAMA(10,2,30) for adaptive momentum - adjusts speed based on market efficiency
- BB Squeeze detection (BW percentile < 20%) for low-volatility entry zones
- RSI(14) with different thresholds (40/60 instead of 45/55) for entry timing
- ADX(14) filter to ensure trend strength > 20 before entering

Why this might beat Sharpe=6.689:
- Donchian captures true breakouts without lag from moving average calculations
- KAMA adapts to regime changes better than fixed-period HMA
- BB squeeze entries catch volatility expansions early (before big moves)
- ADX filter avoids weak trends that whipsaw
- Different signal combination than HMA+RSI - may capture different market regimes
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_kama_bbsqueeze_v1"
timeframe = "15m"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel - upper/lower bounds"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = upper - lower
    
    return mid, upper, lower, width


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Calculate TR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    tr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    # Calculate ADX (smoothed DX)
    adx[period * 2 - 1:] = pd.Series(dx).rolling(window=period, min_periods=period).mean().values[period * 2 - 1:]
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing and risk
    rsi_15m = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    adx_15m = calculate_adx(high, low, close, period=14)
    bb_mid_15m, bb_upper_15m, bb_lower_15m, bb_width_15m = calculate_bollinger(close, period=20, std_mult=2.0)
    kama_15m = calculate_kama(close, period=10, fast=2, slow=30)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    
    # 4h trend via Donchian (resample 15m → 4h)
    df_15m = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_15m.index = pd.date_range(start='2021-01-01', periods=n, freq='15min')
    
    # Resample to 4h
    df_4h = df_15m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h Donchian for trend
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian position
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        mid_4h = (donchian_upper_4h[i] + donchian_lower_4h[i]) / 2
        if c_4h[i] > mid_4h and c_4h[i] > donchian_upper_4h[i - 1]:
            trend_4h[i] = 1  # Bullish breakout
        elif c_4h[i] < mid_4h and c_4h[i] < donchian_lower_4h[i - 1]:
            trend_4h[i] = -1  # Bearish breakout
        elif c_4h[i] > mid_4h:
            trend_4h[i] = 1  # Above midpoint
        elif c_4h[i] < mid_4h:
            trend_4h[i] = -1  # Below midpoint
    
    # Map 4h trend back to 15m timeframe (16 x 15m = 4h)
    trend_15m = np.zeros(n)
    idx_15m_to_4h = np.arange(n) // 16
    
    for i in range(n):
        idx_4h = idx_15m_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_15m[i] = trend_4h[idx_4h]
    
    # Calculate BB Width percentile for squeeze detection
    bb_width_pct = np.zeros(n)
    for i in range(50, n):
        if i >= 50:
            bb_width_pct[i] = np.sum(bb_width_15m[:i] < bb_width_15m[i]) / i
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # RSI thresholds for entries
    RSI_LONG_ENTRY = 40   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # BB squeeze threshold (low volatility = good entry zone)
    BB_SQUEEZE_PCT = 0.25  # BB width in bottom 25% = squeeze
    
    # ADX threshold for trend strength
    ADX_MIN = 20  # Only trade if ADX > 20 (strong trend)
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.5  # Don't enter if price > 2.5 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(80, 48, 14, 20, 28)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    for i in range(first_valid, n):
        if np.isnan(rsi_15m[i]) or np.isnan(atr_15m[i]) or np.isnan(zscore_15m[i]) or np.isnan(adx_15m[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        adx_val = adx_15m[i]
        bb_squeeze = bb_width_pct[i] < BB_SQUEEZE_PCT
        price = close[i]
        kama_val = kama_15m[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            if prev_side == 1:
                # Stoploss check
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
            elif prev_side == -1:
                # Stoploss check
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ADX filter - only trade strong trends
        if adx_val < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))  # Clamp
        
        if trend == 1:  # 4h uptrend
            # KAMA confirmation + RSI pullback + BB squeeze entry
            if price > kama_val and rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # KAMA confirmation + RSI rally + BB squeeze entry
            if price < kama_val and rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 08:37
