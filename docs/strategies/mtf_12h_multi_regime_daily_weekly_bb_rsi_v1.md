# Strategy: mtf_12h_multi_regime_daily_weekly_bb_rsi_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.363 | +0.6% | -18.0% | 447 | FAIL |
| ETHUSDT | -0.191 | +6.1% | -23.2% | 447 | FAIL |
| SOLUSDT | 0.710 | +110.0% | -42.6% | 449 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.240 | +9.8% | -16.5% | 149 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #041: 12h Multi-Regime Strategy with Daily/Weekly HMA + BB Width + RSI
Hypothesis: 12h timeframe captures swing trades while avoiding intraday noise.
Using BOTH 1d and 1w HTF for stronger regime filtering. Bollinger Band Width detects
volatility regime (squeeze=breakout coming, wide=trending). RSI with relaxed thresholds
(25/75 instead of 30/70) ensures sufficient trade generation. Weekly HMA provides
macro bull/bear filter that prevents counter-trend trades in strong regimes.
Position sizing 0.25 with 2.5x ATR stoploss balances risk/reward.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_multi_regime_daily_weekly_bb_rsi_v1"
timeframe = "12h"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Band Width = (Upper - Lower) / SMA
    bw = np.where(sma > 0, (upper - lower) / sma, 0)
    # BB Percentile = (Close - Lower) / (Upper - Lower)
    bb_pct = np.where((upper - lower) > 0, (close - lower) / (upper - lower), 0.5)
    return upper, lower, sma, bw, bb_pct

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    bb_upper, bb_lower, bb_sma, bb_width, bb_pct = calculate_bollinger_bands(close, 20, 2.0)
    
    # 12h HMA for trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # BB Width percentile for regime detection (volatile vs squeeze)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).rank(pct=True).values
    bb_width_pct = np.nan_to_num(bb_width_pct, nan=0.5)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit = 0.0
    
    for i in range(100, n):
        # Weekly macro regime (bull/bear)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 12h Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # 12h HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI signals (relaxed thresholds for more trades)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Bollinger Band regime
        bb_squeeze = bb_width_pct[i] < 0.3  # Low volatility = breakout coming
        bb_expanding = bb_width_pct[i] > 0.7  # High volatility = trending
        bb_lower_touch = bb_pct[i] < 0.1  # Price at lower band
        bb_upper_touch = bb_pct[i] > 0.9  # Price at upper band
        
        # Price position
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (multiple paths to ensure trades)
        # Trigger 1: Supertrend flip long (strongest signal)
        if st_flip_long:
            new_signal = SIZE
        # Trigger 2: Weekly bullish + Daily bullish + Supertrend long + RSI rising
        elif weekly_bullish and daily_bullish and st_long and rsi_rising:
            new_signal = SIZE
        # Trigger 3: BB squeeze + Supertrend long + price above HMA21 (breakout setup)
        elif bb_squeeze and st_long and price_above_hma21:
            new_signal = SIZE
        # Trigger 4: RSI oversold + Supertrend long + weekly bullish (pullback entry)
        elif rsi_oversold and st_long and weekly_bullish:
            new_signal = SIZE
        # Trigger 5: BB lower touch + Supertrend long (mean reversion in uptrend)
        elif bb_lower_touch and st_long:
            new_signal = SIZE
        # Trigger 6: HMA trend long + RSI neutral + Supertrend long (trend continuation)
        elif hma_trend_long and rsi_neutral and st_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Supertrend flip short (strongest signal)
        if st_flip_short:
            new_signal = -SIZE
        # Trigger 2: Weekly bearish + Daily bearish + Supertrend short + RSI falling
        elif weekly_bearish and daily_bearish and st_short and rsi_falling:
            new_signal = -SIZE
        # Trigger 3: BB squeeze + Supertrend short + price below HMA21 (breakdown setup)
        elif bb_squeeze and st_short and price_below_hma21:
            new_signal = -SIZE
        # Trigger 4: RSI overbought + Supertrend short + weekly bearish (pullback entry)
        elif rsi_overbought and st_short and weekly_bearish:
            new_signal = -SIZE
        # Trigger 5: BB upper touch + Supertrend short (mean reversion in downtrend)
        elif bb_upper_touch and st_short:
            new_signal = -SIZE
        # Trigger 6: HMA trend short + RSI neutral + Supertrend short (trend continuation)
        elif hma_trend_short and rsi_neutral and st_short:
            new_signal = -SIZE
        
        # Stoploss and take profit logic (Rule 6)
        if position_side > 0 and entry_price > 0:
            # Initial stoploss
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Track max profit for take profit
                if close[i] > entry_price:
                    max_profit = max(max_profit, close[i] - entry_price)
                # Take partial profit at 2.5R
                if max_profit >= 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Initial stoploss
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                # Track max profit for take profit
                if close[i] < entry_price:
                    max_profit = max(max_profit, entry_price - close[i])
                # Take partial profit at 2.5R
                if max_profit >= 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            max_profit = 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
                max_profit = 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            max_profit = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 01:22
