# Strategy: mtf_4h_crsi_donchian_chop_regime_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.623 | +0.9% | -9.9% | 585 | FAIL |
| ETHUSDT | -0.201 | +11.2% | -13.2% | 598 | FAIL |
| SOLUSDT | 0.330 | +43.1% | -21.3% | 578 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.333 | +10.1% | -14.7% | 199 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #691: 4h Primary + 1d HTF — CRSI Mean Reversion + Donchian Trend + Choppiness Regime

Hypothesis: After 689 failed with Fisher, return to proven CRSI + regime switching.
The key insight from history: regime detection MUST work to switch between mean-revert
and trend-follow. Choppiness Index is the most reliable regime filter for crypto.

Strategy Logic:
1. Choppiness Index (CHOP) detects regime:
   - CHOP > 61.8 = choppy/range → use CRSI mean reversion
   - CHOP < 38.2 = trending → use Donchian breakout
   - Between = neutral → reduce position size or stay flat

2. 1d HMA for trend bias (HTF filter):
   - Price > 1d HMA = only long signals allowed (or stronger long)
   - Price < 1d HMA = only short signals allowed (or stronger short)

3. CRSI (Connors RSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > 1d HMA
   - Short: CRSI > 85 + price < 1d HMA

4. Donchian(20) for trend breakout:
   - Long: price breaks 20-bar high + ADX > 25
   - Short: price breaks 20-bar low + ADX > 25

5. ATR stops at 2.5x, position size 0.25-0.30

Why this should beat 689:
- CRSI has 75% win rate in backtests (proven in #681)
- Choppiness regime filter worked in #681 (Sharpe=0.077)
- Simpler than triple-regime but more adaptive than single-mode
- LOOSE CRSI thresholds (15/85 not 10/90) ensure trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75%+ win rate at extremes (<10 or >90).
    We use <15 and >85 for more trades.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + rsi_period:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    
    # Component 2: Streak RSI (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        up_streak = max(0, streak[i])
        down_streak = abs(min(0, streak[i]))
        total = up_streak + down_streak + 1e-10
        streak_rsi[i] = 100 * up_streak / total
    
    streak_rsi = np.nan_to_num(streak_rsi, nan=50.0)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank of daily returns over 100 periods
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10) * 100
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        current_return = returns[i]
        rank = np.sum(window < current_return) / rank_period * 100
        percent_rank[i] = rank
    
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    percent_rank = np.clip(percent_rank, 0, 100)
    
    # Combine all three components
    crsi = (rsi + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - detects trending vs ranging markets.
    CHOP > 61.8 = choppy/range (mean reversion works)
    CHOP < 38.2 = trending (breakout works)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate CHOP
    for i in range(period - 1, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels - breakout detection."""
    n = len(close) if 'close' in dir() else len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF (1d) HMA for trend bias
    def calculate_hma(series, period):
        """Half-weighted Moving Average."""
        if len(series) < period:
            return np.full(len(series), np.nan)
        wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
        wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
        hma_raw = 2 * wma1 - wma2
        hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
        return hma
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 61.8
        is_trending = chop_4h[i] < 38.2
        is_neutral = not is_choppy and not is_trending
        
        # === TREND BIAS (1d HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === ADX STRENGTH ===
        adx_strong = adx_4h[i] > 25
        
        desired_signal = 0.0
        
        # === MEAN REVERSION MODE (Choppy Market) ===
        if is_choppy:
            # Long: CRSI oversold + trend bullish bias
            if crsi_4h[i] < 15 and trend_bullish:
                desired_signal = SIZE_LONG
            
            # Short: CRSI overbought + trend bearish bias
            elif crsi_4h[i] > 85 and trend_bearish:
                desired_signal = -SIZE_SHORT
            
            # Weaker signals in neutral trend
            elif crsi_4h[i] < 10 and not trend_bearish:
                desired_signal = SIZE_LONG * 0.5
            elif crsi_4h[i] > 90 and not trend_bullish:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === TREND FOLLOWING MODE (Trending Market) ===
        elif is_trending:
            # Long breakout: price breaks Donchian high + ADX strong + trend bullish
            if close[i] > donchian_upper_4h[i-1] and adx_strong and trend_bullish:
                desired_signal = SIZE_LONG
            
            # Short breakout: price breaks Donchian low + ADX strong + trend bearish
            elif close[i] < donchian_lower_4h[i-1] and adx_strong and trend_bearish:
                desired_signal = -SIZE_SHORT
            
            # Weaker breakout without ADX confirmation
            elif close[i] > donchian_upper_4h[i-1] and trend_bullish:
                desired_signal = SIZE_LONG * 0.5
            elif close[i] < donchian_lower_4h[i-1] and trend_bearish:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === NEUTRAL MODE (Reduce exposure) ===
        elif is_neutral:
            # Only take extreme CRSI signals at half size
            if crsi_4h[i] < 10 and trend_bullish:
                desired_signal = SIZE_LONG * 0.5
            elif crsi_4h[i] > 90 and trend_bearish:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and trend still bullish
                if crsi_4h[i] < 80 and trend_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend still bearish
                if crsi_4h[i] > 20 and trend_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR trend reverses below 1d HMA
        if in_position and position_side > 0:
            if crsi_4h[i] > 80 or (close[i] < hma_1d_aligned[i] and chop_4h[i] < 50):
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR trend reverses above 1d HMA
        if in_position and position_side < 0:
            if crsi_4h[i] < 20 or (close[i] > hma_1d_aligned[i] and chop_4h[i] < 50):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
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
2026-03-23 12:47
