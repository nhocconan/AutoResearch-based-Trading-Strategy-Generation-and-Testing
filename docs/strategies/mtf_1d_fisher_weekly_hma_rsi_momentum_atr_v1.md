# Strategy: mtf_1d_fisher_weekly_hma_rsi_momentum_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.281 | +1.5% | -29.4% | 249 | FAIL |
| ETHUSDT | -0.704 | -27.6% | -45.5% | 253 | FAIL |
| SOLUSDT | 0.896 | +171.3% | -27.1% | 255 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.377 | +12.7% | -17.4% | 84 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #408: 1d Fisher Transform + Weekly HMA Bias + RSI Momentum + ATR Stop
Hypothesis: Fisher Transform normalizes price distribution and identifies extreme reversal points
better than standard oscillators. Combined with weekly HMA for trend bias and RSI for momentum
confirmation, this should generate MORE trades than previous 1d strategies (which failed with 0 trades).
Key changes from #396: Simpler entry logic (fewer filters), Fisher crossovers trigger entries more
frequently than KAMA crossovers, relaxed RSI thresholds (25-75 instead of 35-65), weekly HMA as
simple directional bias only (not strict filter). Target: Beat Sharpe=0.499 with >=10 trades/symbol.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Position size: 0.30 discrete, stoploss 2.5*ATR for daily timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_weekly_hma_rsi_momentum_atr_v1"
timeframe = "1d"
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

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    # Calculate highest high and lowest low over period
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1]) if 'high' in dir() else np.max(close[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1]) if 'low' in dir() else np.min(close[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        if highest == lowest:
            continue
        
        normalized = 2 * (close[i] - lowest) / (highest - lowest) - 1
        
        # Apply exponential smoothing
        if i == period:
            smoothed = normalized
        else:
            smoothed = 0.67 * normalized + 0.33 * smoothed_prev
        
        smoothed_prev = smoothed
        
        # Clamp to avoid division by zero
        smoothed = np.clip(smoothed, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + smoothed) / (1 - smoothed))
        
        # Trigger line (1-period lag)
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_fisher_simple(close, period=9):
    """Simplified Fisher Transform using only close price."""
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Use rolling high/low from close
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            if i > period:
                trigger[i] = fisher[i-1]
            continue
        
        # Normalize
        normalized = 2.0 * (close[i] - lowest) / (highest - lowest) - 1.0
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Trigger (lagged fisher)
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, trigger = calculate_fisher_simple(close, 9)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (long-term direction) - SOFT filter
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # SMA50 trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # Fisher Transform signals (reversal detection)
        fisher_bull_cross = fisher[i] > -1.5 and trigger[i] <= -1.5  # Cross above -1.5
        fisher_bear_cross = fisher[i] < 1.5 and trigger[i] >= 1.5   # Cross below +1.5
        
        # Fisher extreme levels (oversold/overbought)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Fisher turning up/down
        fisher_turning_up = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_turning_down = fisher[i] < fisher[i-1] if i > 0 else False
        
        # RSI momentum (RELAXED thresholds to ensure trades)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 80  # Wide range
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 70  # Wide range
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Fisher cross + Weekly bullish + RSI ok (primary)
        if fisher_bull_cross and weekly_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Fisher oversold + turning up + above SMA50
        elif fisher_oversold and fisher_turning_up and above_sma50 and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Path 3: Weekly bullish + Fisher turning up + RSI momentum
        elif weekly_bullish and fisher_turning_up and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 4: Fisher cross + above SMA50 (weekly neutral ok)
        elif fisher_bull_cross and above_sma50 and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: Simple momentum - price > SMA50 + Fisher > 0 + RSI > 50
        elif above_sma50 and fisher[i] > 0 and rsi[i] > 50 and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Fisher cross + Weekly bearish + RSI ok (primary)
        if fisher_bear_cross and weekly_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Fisher overbought + turning down + below SMA50
        elif fisher_overbought and fisher_turning_down and below_sma50 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 3: Weekly bearish + Fisher turning down + RSI momentum
        elif weekly_bearish and fisher_turning_down and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 4: Fisher cross + below SMA50 (weekly neutral ok)
        elif fisher_bear_cross and below_sma50 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple momentum - price < SMA50 + Fisher < 0 + RSI < 50
        elif below_sma50 and fisher[i] < 0 and rsi[i] < 50 and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for daily timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest for daily timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 05:56
