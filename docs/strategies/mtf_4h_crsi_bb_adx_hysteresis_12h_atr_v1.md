# Strategy: mtf_4h_crsi_bb_adx_hysteresis_12h_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.356 | +4.1% | -12.7% | 478 | FAIL |
| ETHUSDT | -0.032 | +17.0% | -17.6% | 479 | FAIL |
| SOLUSDT | 1.014 | +147.4% | -18.5% | 451 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.056 | +6.1% | -11.9% | 136 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #764: 4h Primary + 12h HTF — Connors RSI + Bollinger Mean Reversion + ADX Hysteresis

Hypothesis: After analyzing 500+ failed strategies and the current best (Sharpe=0.612):
1. Connors RSI (CRSI) has 75% win rate in mean reversion but needs simpler thresholds
2. Bollinger Bands (20, 2.0) provide clearer mean reversion bounds than Keltner
3. ADX hysteresis (enter >25, exit <18) reduces regime whipsaw vs single threshold
4. 12h EMA50 is better trend filter than 1d for 4h entries (faster response)
5. Volume filter too strict in #761 — relax to 1.2x for more trades
6. Simpler hold logic: maintain position until opposite signal or stoploss

Strategy design:
1. 12h EMA(50) for trend bias (aligned via mtf_data helper)
2. 4h Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. 4h Bollinger Bands (20, 2.0) for mean reversion bounds
4. 4h ADX(14) with hysteresis for regime (trending >25, ranging <18)
5. 4h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key improvements from #761:
- Replaced Keltner with Bollinger (more proven for crypto mean reversion)
- Replaced RSI(14) with Connors RSI (faster, more responsive)
- Added ADX hysteresis (reduces regime flip-flop)
- Relaxed volume filter (1.2x vs 1.5x) for more trade frequency
- Simpler hold/exit logic (less overfitting)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_bb_adx_hysteresis_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    Connors RSI Streak component.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    # Calculate streak values
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (inverted for down streaks)
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    
    # Calculate RSI of streak scores
    streak_rsi = calculate_rsi(streak_score + 100, period)  # shift to positive
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <10 = oversold, >90 = overbought.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    # Handle NaN values
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    adx_4h = calculate_adx(high, low, close, period=14)
    vol_sma_4h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF EMA for trend bias
    ema_12h_raw = calculate_ema(df_12h['close'].values, 50)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trending, 2=ranging
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(ema_12h_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(adx_4h[i]):
            continue
        if np.isnan(vol_sma_4h[i]) or vol_sma_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (12h HTF EMA50) ===
        trend_12h_bullish = close[i] > ema_12h_aligned[i]
        trend_12h_bearish = close[i] < ema_12h_aligned[i]
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        # Enter trending when ADX > 25, exit when ADX < 18
        if adx_4h[i] > 25:
            adx_regime = 1  # trending
        elif adx_4h[i] < 18:
            adx_regime = 2  # ranging
        else:
            adx_regime = prev_adx_regime  # maintain previous
        
        prev_adx_regime = adx_regime
        trending_regime = (adx_regime == 1)
        ranging_regime = (adx_regime == 2)
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 1.2 * vol_sma_4h[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 15
        crsi_overbought = crsi_4h[i] > 85
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        near_bb_lower = close[i] < bb_sma[i] and close[i] > bb_lower[i] * 0.995
        near_bb_upper = close[i] > bb_sma[i] and close[i] < bb_upper[i] * 1.005
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (ADX < 18) ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + below BB lower + trend not bearish
            if crsi_oversold and below_bb_lower and not trend_12h_bearish:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + above BB upper + trend not bullish
            if crsi_overbought and above_bb_upper and not trend_12h_bullish:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Conservative: extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_12h_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_12h_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (ADX > 25) ===
        elif trending_regime:
            # Trend pullback long: 12h bullish + CRSI neutral + near BB SMA
            if trend_12h_bullish and 35 < crsi_4h[i] < 50:
                if near_bb_lower or close[i] < bb_sma[i]:
                    desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 12h bearish + CRSI neutral + near BB SMA
            if trend_12h_bearish and 50 < crsi_4h[i] < 65:
                if near_bb_upper or close[i] > bb_sma[i]:
                    desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Breakout continuation with volume
            if trend_12h_bullish and above_bb_upper and volume_confirmed:
                desired_signal = BASE_SIZE
            
            if trend_12h_bearish and below_bb_lower and volume_confirmed:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (18 <= ADX <= 25) ===
        else:
            # Conservative: only extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_12h_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_12h_bearish:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if trend_12h_bullish and crsi_4h[i] < 80:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_12h_bearish and crsi_4h[i] > 20:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI overbought
            if trend_12h_bearish and crsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI oversold
            if trend_12h_bullish and crsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
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
2026-03-23 14:15
