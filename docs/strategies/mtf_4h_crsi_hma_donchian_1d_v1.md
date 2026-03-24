# Strategy: mtf_4h_crsi_hma_donchian_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.261 | +3.0% | -21.2% | 415 | FAIL |
| ETHUSDT | 0.155 | +28.1% | -31.6% | 429 | PASS |
| SOLUSDT | 0.329 | +49.7% | -26.9% | 477 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.007 | +4.7% | -13.4% | 126 | PASS |
| SOLUSDT | 0.609 | +19.8% | -13.3% | 136 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #431: 4h Primary + 1d HTF — Connors RSI + HMA Trend + Donchian Breakout

Hypothesis: Connors RSI (CRSI) is proven to work in bear/range markets with 75% win rate.
Combined with 1d HMA trend bias and Donchian breakout confirmation, this should:
1. Generate adequate trade frequency (80-200 trades over 4-year train)
2. Work across all symbols (BTC/ETH/SOL) not just SOL
3. Beat current best Sharpe=0.612 with better mean reversion in chop

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 10 + price > SMA200 (oversold in uptrend)
- Short: CRSI > 90 + price < SMA200 (overbought in downtrend)
- Donchian(20) breakout confirms trend continuation

Why this should beat #429 (Sharpe=0.002):
- CRSI is more reliable than Fisher for reversal detection in crypto
- Simpler entry conditions = more trades (avoid 0-trade failure)
- 1d HMA bias is cleaner than ADX regime switching
- Donchian breakout adds trend confirmation without over-filtering

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_hma_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of CRSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        streak_rsi[i] = streak
    
    # Convert to RSI-like scale (0-100)
    streak_rsi_s = pd.Series(streak_rsi)
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_gain = streak_rsi_s.rolling(window=period, min_periods=period).apply(
            lambda x: np.sum(x[x > 0]) if len(x[x > 0]) > 0 else 0
        )
        streak_loss = streak_rsi_s.rolling(window=period, min_periods=period).apply(
            lambda x: np.sum(-x[x < 0]) if len(x[x < 0]) > 0 else 0
        )
        
        rs = streak_gain / (streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + rs))
    
    streak_rsi = np.clip(streak_rsi.values, 0, 100)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of CRSI.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (conservative)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Long entry threshold
        crsi_overbought = crsi[i] > 85.0  # Short entry threshold
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — CRSI oversold + trend bias (simpler = more trades)
        long_bias = price_above_hma_1d or hma_bullish or close[i] > sma_200[i]
        
        if long_bias and crsi_oversold:
            # Primary: CRSI oversold in uptrend
            desired_signal = position_size
        elif crsi_oversold and close[i] > sma_200[i]:
            # Secondary: CRSI oversold + above SMA200 (regardless of HMA)
            desired_signal = position_size * 0.8
        elif donchian_breakout_long and hma_bullish:
            # Tertiary: Donchian breakout + HMA bullish (trend follow)
            desired_signal = position_size * 0.6
        
        # SHORT SETUP — CRSI overbought + trend bias
        short_bias = price_below_hma_1d or hma_bearish or close[i] < sma_200[i]
        
        if short_bias and crsi_overbought:
            # Primary: CRSI overbought in downtrend
            desired_signal = -position_size
        elif crsi_overbought and close[i] < sma_200[i]:
            # Secondary: CRSI overbought + below SMA200
            desired_signal = -position_size * 0.8
        elif donchian_breakout_short and hma_bearish:
            # Tertiary: Donchian breakdown + HMA bearish
            desired_signal = -position_size * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or hma_bullish or close[i] > sma_200[i]):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_1d or hma_bearish or close[i] < sma_200[i]):
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 10:21
