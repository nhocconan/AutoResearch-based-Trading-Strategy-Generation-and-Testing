# Strategy: mtf_1d_donchian_weekly_hma_rsi_breakout_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.347 | -0.9% | -23.3% | 114 | FAIL |
| ETHUSDT | -0.198 | +3.4% | -28.8% | 118 | FAIL |
| SOLUSDT | 0.847 | +149.1% | -21.1% | 105 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.506 | +14.0% | -8.5% | 16 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #438: 1d Donchian Breakout + Weekly HMA Bias + RSI Filter + ATR Stop
Hypothesis: Donchian channel breakouts (Turtle Trading style) work well on daily timeframes
for capturing major trends. Weekly HMA provides higher timeframe bias to filter false breakouts.
RSI filter avoids entering at extreme overbought/oversold levels. ATR-based stoploss (3*ATR)
controls drawdown on daily timeframe. Multiple entry paths ensure >=10 trades per symbol.
Position size: 0.30 discrete, stoploss 3*ATR for daily timeframe (wider than intraday).
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_rsi_breakout_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (Turtle Trading breakout system).
    Upper = highest high over N periods
    Lower = lowest low over N periods
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    sma50 = calculate_sma(close, 50)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (long-term direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # ADX trend strength filter (only trade when ADX > 20)
        trend_strength = adx[i] > 20
        
        # RSI filter (avoid extremes)
        rsi_not_overbought = rsi[i] < 75
        rsi_not_oversold = rsi[i] > 25
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 60
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # DI crossover signals
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Previous day's position in Donchian channel
        prev_in_channel = True
        if i > 1 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            prev_in_channel = donchian_lower[i-1] <= close[i-1] <= donchian_upper[i-1]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Weekly bullish + ADX trend + RSI not overbought
        if breakout_long and weekly_bullish and trend_strength and rsi_not_overbought:
            new_signal = SIZE_ENTRY
        # Path 2: Donchian breakout + Above SMA50 + DI bullish + RSI neutral
        elif breakout_long and above_sma50 and di_bullish and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 3: Price near upper Donchian + Weekly bullish + ADX > 25
        elif close[i] > donchian_upper[i] * 0.98 and weekly_bullish and adx[i] > 25 and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 4: Breakout from channel + Weekly bullish + RSI 40-65
        elif breakout_long and weekly_bullish and rsi[i] > 40 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        # Path 5: DI bullish + Weekly bullish + Above SMA50 + RSI > 45
        elif di_bullish and weekly_bullish and above_sma50 and rsi[i] > 45 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Path 6: Price above SMA50 + Weekly bullish + ADX rising
        elif above_sma50 and weekly_bullish and adx[i] > adx[i-1] and adx[i] > 18 and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Weekly bearish + ADX trend + RSI not oversold
        if breakout_short and weekly_bearish and trend_strength and rsi_not_oversold:
            new_signal = -SIZE_ENTRY
        # Path 2: Donchian breakout + Below SMA50 + DI bearish + RSI neutral
        elif breakout_short and below_sma50 and di_bearish and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Price near lower Donchian + Weekly bearish + ADX > 25
        elif close[i] < donchian_lower[i] * 1.02 and weekly_bearish and adx[i] > 25 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 4: Breakout from channel + Weekly bearish + RSI 35-60
        elif breakout_short and weekly_bearish and rsi[i] > 35 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: DI bearish + Weekly bearish + Below SMA50 + RSI < 55
        elif di_bearish and weekly_bearish and below_sma50 and rsi[i] > 30 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 6: Price below SMA50 + Weekly bearish + ADX rising
        elif below_sma50 and weekly_bearish and adx[i] > adx[i-1] and adx[i] > 18 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR from highest for daily timeframe)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR from lowest for daily timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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
2026-03-22 06:30
