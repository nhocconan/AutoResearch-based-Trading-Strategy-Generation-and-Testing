# Strategy: mtf_12h_kama_fisher_1d_hma_regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.386 | +0.9% | -18.4% | 258 | FAIL |
| ETHUSDT | -0.502 | -11.6% | -25.6% | 254 | FAIL |
| SOLUSDT | 0.577 | +86.7% | -20.8% | 227 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.278 | +10.7% | -12.4% | 78 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #017: 12h KAMA + Fisher Transform with 1d HMA Regime Filter
Hypothesis: KAMA adapts to volatility better than EMA/HMA, reducing whipsaw in 2022 crash.
Fisher Transform catches reversals more reliably than RSI in bear markets (research-backed).
1d HMA provides regime bias (bull/bear) to filter counter-trend trades.
Volume confirmation ensures breakouts have participation.
Asymmetric sizing: smaller positions in bear regime to limit drawdown.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.25 base, 0.30 in bull regime, 0.20 in bear regime.
Stoploss: 2.5*ATR trailing stop.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_1d_hma_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    Less whipsaw than EMA in ranging markets.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Better at identifying turning points than RSI in bear markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Calculate highest high and lowest low over period
    for i in range(period - 1, n):
        hh = np.max(close[i - period + 1:i + 1])
        ll = np.min(close[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize price to range 0-1
            normalized = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * (fisher_signal[i - 1] if i > period - 1 else 0.0)
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            if i > period:
                fisher_signal[i] = fisher[i - 1]
            else:
                fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_ma
    ratio[np.isnan(ratio)] = 1.0
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - asymmetric based on regime
    SIZE_BULL = 0.30  # Larger in bull regime
    SIZE_BEAR = 0.20  # Smaller in bear regime (risk management)
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        # 1d regime bias (HTF) - determines which direction to favor
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # KAMA trend direction
        kama_rising = kama[i] > kama[i - 5] if i > 5 else False
        kama_falling = kama[i] < kama[i - 5] if i > 5 else False
        
        # Price position vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] < -1.5  # Cross above -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] > 1.5   # Cross below +1.5
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Select position size based on regime
        current_size = SIZE_BULL if bull_regime else SIZE_BEAR
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: KAMA rising + Fisher long cross + bull regime + volume
        if kama_rising and fisher_long and bull_regime and volume_confirmed:
            new_signal = current_size
        # Secondary: Fisher oversold + price above KAMA + bull regime (pullback entry)
        elif fisher_oversold and price_above_kama and bull_regime:
            new_signal = current_size
        # Tertiary: EMA bullish + KAMA rising + bull regime (trend continuation)
        elif ema_bullish and kama_rising and bull_regime:
            new_signal = current_size
        
        # === SHORT ENTRY ===
        # Primary: KAMA falling + Fisher short cross + bear regime + volume
        if kama_falling and fisher_short and bear_regime and volume_confirmed:
            new_signal = -current_size
        # Secondary: Fisher overbought + price below KAMA + bear regime (pullback entry)
        elif fisher_overbought and price_below_kama and bear_regime:
            new_signal = -current_size
        # Tertiary: EMA bearish + KAMA falling + bear regime (trend continuation)
        elif ema_bearish and kama_falling and bear_regime:
            new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 08:43
