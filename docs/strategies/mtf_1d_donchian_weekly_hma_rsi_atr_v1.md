# Strategy: mtf_1d_donchian_weekly_hma_rsi_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.162 | +6.6% | -19.4% | 142 | FAIL |
| ETHUSDT | -0.178 | +1.3% | -24.4% | 150 | FAIL |
| SOLUSDT | 0.931 | +201.2% | -25.4% | 153 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.216 | +33.9% | -11.5% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #306: 1d Donchian Breakout + Weekly HMA Bias + RSI Confirmation + ATR Stops
Hypothesis: Daily Donchian breakouts (20-period) capture major trend moves while weekly HMA 
provides macro directional bias. RSI confirmation (40-70 for long, 30-60 for short) filters 
false breakouts. This is simpler than previous HMA crossover approaches and should generate 
more trades while maintaining quality. ATR trailing stops (2.5*ATR) control drawdown.
Position size 0.30 balances returns vs risk. Target: Beat Sharpe=0.499 with >=10 trades/symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_rsi_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Track previous values for breakout detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # RSI confirmation zones (generous to ensure trades)
        rsi_bullish = 35 < rsi[i] < 70
        rsi_bearish = 30 < rsi[i] < 65
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 55
        
        # Donchian breakout signals
        breakout_long = prev_high[i] <= donchian_upper[i] and close[i] > donchian_upper[i]
        breakout_short = prev_low[i] >= donchian_lower[i] and close[i] < donchian_lower[i]
        
        # Price near Donchian bands (potential breakout)
        near_upper = close[i] > donchian_upper[i] * 0.98
        near_lower = close[i] < donchian_lower[i] * 1.02
        
        # Trend continuation (price above/below Donchian mid)
        donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
        above_mid = close[i] > donchian_mid
        below_mid = close[i] < donchian_mid
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bullish + Donchian breakout + RSI confirmation
        if weekly_bullish and breakout_long and rsi_bullish:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + Above SMA50 + Near upper + RSI momentum
        elif weekly_bullish and above_sma50 and near_upper and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian breakout + Above mid + RSI > 40 (simpler for more trades)
        elif breakout_long and above_mid and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Quaternary: Above SMA50 + Above Donchian mid + RSI 45-65 (trend continuation)
        elif above_sma50 and above_mid and 45 < rsi[i] < 65:
            new_signal = SIZE_ENTRY
        # Simple: Weekly bullish + Price > SMA50 + RSI > 50
        elif weekly_bullish and above_sma50 and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Weekly bearish + Donchian breakout + RSI confirmation
        if weekly_bearish and breakout_short and rsi_bearish:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + Below SMA50 + Near lower + RSI momentum
        elif weekly_bearish and below_sma50 and near_lower and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian breakout + Below mid + RSI < 60 (simpler for more trades)
        elif breakout_short and below_mid and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: Below SMA50 + Below Donchian mid + RSI 35-55 (trend continuation)
        elif below_sma50 and below_mid and 35 < rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Simple: Weekly bearish + Price < SMA50 + RSI < 50
        elif weekly_bearish and below_sma50 and rsi[i] < 50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
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
            
            # Calculate trailing stop (2.5*ATR from lowest)
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
2026-03-22 04:36
