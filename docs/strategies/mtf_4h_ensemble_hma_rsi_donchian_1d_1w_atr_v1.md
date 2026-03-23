# Strategy: mtf_4h_ensemble_hma_rsi_donchian_1d_1w_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.167 | +25.6% | -4.4% | 552 | PASS |
| ETHUSDT | -1.086 | -4.3% | -8.0% | 538 | FAIL |
| SOLUSDT | 0.226 | +31.7% | -10.4% | 606 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.075 | -5.4% | -6.6% | 235 | FAIL |
| SOLUSDT | 0.140 | +7.3% | -5.4% | 214 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #256: 4h Multi-Signal Ensemble with 1d/1w HMA Bias

Hypothesis: 4h timeframe captures swing trades (2-10 day holds) with good signal quality.
Using ensemble of 3 independent signals (HMA crossover, RSI pullback, Donchian breakout)
with 1d/1w HMA for directional bias. Each signal votes independently - need 2/3 for entry.

Why this might work:
- 4h is the "sweet spot" - less noise than 1h, more trades than 1d
- Ensemble approach reduces false signals from any single indicator
- 1d HMA provides intermediate trend bias (not too restrictive)
- 1w HMA provides macro bias (only used for conviction, not hard filter)
- RSI(7) pullback entries in trend direction = high win rate
- Donchian breakout captures momentum continuation
- Conservative sizing (0.25) + ATR stoploss controls drawdown

Key improvements over failed experiments:
- #250 (4h sentiment): Sharpe=0.000 - 0 trades, conditions too strict
- #244 (4h fisher): Sharpe=0.000 - 0 trades
- This uses LOOSER entry thresholds to ensure trades happen
- Multiple entry paths (HMA cross OR RSI pullback OR Donchian) = more trades
- 1d HMA bias is soft (increases size, doesn't block entry)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ensemble_hma_rsi_donchian_1d_1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_9 = calculate_hma(close, 9)
    hma_21 = calculate_hma(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_price_idx = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_9[i]) or np.isnan(hma_21[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = intermediate trend bias (soft filter)
        # 1w HMA = macro trend bias (conviction booster)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === SIGNAL 1: HMA CROSSOVER ===
        hma_cross_long = hma_9[i] > hma_21[i] and hma_9[i-1] <= hma_21[i-1]
        hma_cross_short = hma_9[i] < hma_21[i] and hma_9[i-1] >= hma_21[i-1]
        
        # HMA alignment (both trending same direction)
        hma_aligned_long = hma_9[i] > hma_21[i] and hma_9[i] > hma_9[i-3]
        hma_aligned_short = hma_9[i] < hma_21[i] and hma_9[i] < hma_9[i-3]
        
        # === SIGNAL 2: RSI PULLBACK ===
        # Long: RSI pulled back to 35-45 in uptrend
        rsi_pullback_long = 35 <= rsi_7[i] <= 50 and close[i] > ema_50[i]
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_short = 50 <= rsi_7[i] <= 65 and close[i] < ema_50[i]
        
        # RSI extreme reversal (mean reversion)
        rsi_oversold = rsi_7[i] < 30
        rsi_overbought = rsi_7[i] > 70
        
        # === SIGNAL 3: DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation for breakouts
        volume_confirmed = volume[i] > 1.5 * vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        # === ENSEMBLE VOTING ===
        long_votes = 0
        short_votes = 0
        
        # HMA crossover vote
        if hma_cross_long or hma_aligned_long:
            long_votes += 1
        if hma_cross_short or hma_aligned_short:
            short_votes += 1
        
        # RSI pullback vote
        if rsi_pullback_long:
            long_votes += 1
        if rsi_pullback_short:
            short_votes += 1
        
        # Donchian breakout vote (needs volume confirmation)
        if donchian_breakout_long and volume_confirmed:
            long_votes += 1
        if donchian_breakout_short and volume_confirmed:
            short_votes += 1
        
        # RSI extreme reversal vote (counter-trend, lower weight)
        if rsi_oversold and close[i] > ema_50[i]:
            long_votes += 0.5
        if rsi_overbought and close[i] < ema_50[i]:
            short_votes += 0.5
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # Need at least 1.5 votes for entry (allows single strong signal + HTF bias)
        # Or 2+ votes regardless of HTF
        if long_votes >= 2.0 or (long_votes >= 1.5 and bull_trend_1d):
            new_signal = SIZE_BASE
        
        if short_votes >= 2.0 or (short_votes >= 1.5 and bear_trend_1d):
            new_signal = -SIZE_BASE
        
        # Boost size if 1w agrees (stronger conviction)
        if new_signal > 0 and bull_trend_1w:
            new_signal = min(new_signal + 0.05, 0.30)
        if new_signal < 0 and bear_trend_1w:
            new_signal = max(new_signal - 0.05, -0.30)
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 14:31
