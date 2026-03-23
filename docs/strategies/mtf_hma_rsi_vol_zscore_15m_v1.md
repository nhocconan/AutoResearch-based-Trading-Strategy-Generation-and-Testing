# Strategy: mtf_hma_rsi_vol_zscore_15m_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.835 | -47.6% | -49.9% | 4868 | FAIL |
| ETHUSDT | -12.051 | -94.9% | -95.0% | 13411 | FAIL |
| SOLUSDT | 0.029 | +19.0% | -21.0% | 141 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.831 | +18.1% | -6.9% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 15m primary with 4h HMA trend + 1h RSI pullback + Volume/Z-score filter
- 4h HMA provides stable trend direction (less noise than 15m)
- 1h RSI identifies pullback entries in trend direction
- Volume spike confirms breakout validity
- Z-score detects extreme moves for mean reversion exits
- Asymmetric sizing: larger in strong HTF trend, smaller in weak
- ATR trailing stop for risk management
Timeframe: 15m (primary), 4h + 1h (HTF filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_vol_zscore_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_zscore(close, period=20):
    """Z-score for mean reversion detection"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_sma + 1e-10)
    return ratio

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # 1h RSI for pullback entries
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # 15m indicators
    hma_15m_16 = calculate_hma(close, 16)
    hma_15m_48 = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # BB Width for regime detection
    bb_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_width = (bb_sma + 2*bb_std - (bb_sma - 2*bb_std)) / bb_sma
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x) if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile = np.nan_to_num(bb_width_percentile, 0.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # Base position size 25%
    SIZE_MAX = 0.35   # Max position size 35%
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (4h HMA)
        htf_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i] and close[i] > hma_4h_21_aligned[i]
        htf_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i] and close[i] < hma_4h_21_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # 4h trend strength
        hma_slope_4h = (hma_4h_21_aligned[i] - hma_4h_21_aligned[i-4]) / (hma_4h_21_aligned[i-4] + 1e-10)
        htf_strength = min(abs(hma_slope_4h) * 100, 2.0)  # Cap at 2.0
        
        # 15m trend
        trend_15m = 1.0 if hma_15m_16[i] > hma_15m_48[i] else -1.0
        
        # ADX for trend strength
        adx_strong = adx[i] > 25
        adx_weak = adx[i] < 20
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.2
        
        # Z-score extremes
        z_extreme_long = zscore[i] < -1.5
        z_extreme_short = zscore[i] > 1.5
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop or close[i] < entry_price - atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop or close[i] > entry_price + atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        
        # Determine position size based on HTF strength
        if htf_bull or htf_bear:
            size = min(SIZE_BASE + htf_strength * 0.05, SIZE_MAX)
        else:
            size = SIZE_BASE * 0.5  # Reduce size in neutral regime
        
        # Entry logic - asymmetric based on HTF regime
        if htf_bull:  # Bull regime - prefer longs
            # Trend continuation entry
            if trend_15m > 0 and rsi_15m[i] < 55 and rsi_15m[i] > 35:
                if vol_confirmed or adx_strong:
                    signals[i] = size
                    if prev_signal == 0:
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = close[i]
            # Pullback entry (RSI from 1h aligned)
            elif trend_15m > 0 and rsi_1h_aligned[i] < 50 and rsi_1h_aligned[i] > 30:
                signals[i] = size * 0.8
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Z-score mean reversion exit
            elif z_extreme_short and position_side == 1:
                signals[i] = size * 0.5  # Reduce position
            # Overbought - exit
            elif rsi_15m[i] > 70:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        elif htf_bear:  # Bear regime - prefer shorts
            # Trend continuation entry
            if trend_15m < 0 and rsi_15m[i] > 45 and rsi_15m[i] < 65:
                if vol_confirmed or adx_strong:
                    signals[i] = -size
                    if prev_signal == 0:
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = close[i]
            # Pullback entry
            elif trend_15m < 0 and rsi_1h_aligned[i] > 50 and rsi_1h_aligned[i] < 70:
                signals[i] = -size * 0.8
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Z-score mean reversion exit
            elif z_extreme_long and position_side == -1:
                signals[i] = -size * 0.5  # Reduce position
            # Oversold - exit
            elif rsi_15m[i] < 30:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        else:  # Neutral regime - mean reversion only
            if z_extreme_long and adx_weak:
                signals[i] = -size * 0.5
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            elif z_extreme_short and adx_weak:
                signals[i] = size * 0.5
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif abs(zscore[i]) < 0.5:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
        
        # Discretize signal to reduce churn
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            signals[i] = min(max(round(signals[i] / 0.05) * 0.05, 0.15), SIZE_MAX)
        else:
            signals[i] = max(min(round(signals[i] / 0.05) * 0.05, -0.15), -SIZE_MAX)
        
        prev_signal = signals[i]
    
    return signals
```

## Last Updated
2026-03-22 00:34
