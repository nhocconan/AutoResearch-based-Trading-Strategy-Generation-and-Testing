# Strategy: mtf_12h_supertrend_daily_crsi_donchian_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.609 | -11.2% | -18.8% | 362 | FAIL |
| ETHUSDT | 0.208 | +32.1% | -21.9% | 371 | PASS |
| SOLUSDT | 1.185 | +247.3% | -26.0% | 347 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.332 | -1.2% | -19.5% | 112 | FAIL |
| SOLUSDT | 0.403 | +13.4% | -13.8% | 114 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #035: 12h Supertrend + Daily HMA Regime + Connors RSI + Donchian Breakout
Hypothesis: 12h timeframe captures multi-day swings with less noise than intraday.
Daily HMA provides major bull/bear regime filter. Connors RSI (CRSI) gives superior
entry timing vs standard RSI by incorporating streak and percentile rank.
Donchian breakout provides trend continuation entries when Supertrend already aligned.
Multiple entry triggers (Supertrend flip, CRSI extreme, Donchian breakout) ensure ≥10 trades.
Position sizing 0.28 with 2.5x ATR stoploss protects against crashes while capturing trends.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_daily_crsi_donchian_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long opportunity)
    CRSI > 90 = overbought (short opportunity)
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        pos_streaks = np.sum(streak[max(0, i-streak_period):i] > 0)
        streak_rsi[i] = (pos_streaks / streak_period) * 100 if streak_period > 0 else 50
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < close[i]) / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Donchian channels for breakout detection
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    
    # 12h HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # Daily trend filter (major regime)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 12h Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals (strongest entry trigger)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Connors RSI extremes (mean reversion within trend)
        crsi_oversold = crsi[i] < 25  # Relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 75  # Relaxed from 90 for more trades
        crsi_rising = crsi[i] > crsi[i-3] if i > 3 else True
        crsi_falling = crsi[i] < crsi[i-3] if i > 3 else True
        
        # Donchian breakout signals
        donch_breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        donch_breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # Standard RSI momentum
        rsi_bullish = rsi[i] > 45 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 55
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Price position vs HMA21
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Supertrend flip long with daily support
        if st_flip_long and (daily_bullish or crsi_oversold):
            new_signal = SIZE
        # Trigger 2: Supertrend long + HMA trend + CRSI rising (trend continuation)
        elif st_long and hma_trend_long and crsi_rising and price_above_hma:
            new_signal = SIZE
        # Trigger 3: Daily bullish + Supertrend long + Donchian breakout
        elif daily_bullish and st_long and donch_breakout_long:
            new_signal = SIZE
        # Trigger 4: CRSI oversold + Supertrend long + volume (mean reversion in uptrend)
        elif crsi_oversold and st_long and vol_confirm:
            new_signal = SIZE
        # Trigger 5: RSI bullish + Supertrend long + HMA aligned
        elif rsi_bullish and st_long and hma_trend_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Supertrend flip short with daily resistance
        if st_flip_short and (daily_bearish or crsi_overbought):
            new_signal = -SIZE
        # Trigger 2: Supertrend short + HMA trend + CRSI falling (trend continuation)
        elif st_short and hma_trend_short and crsi_falling and price_below_hma:
            new_signal = -SIZE
        # Trigger 3: Daily bearish + Supertrend short + Donchian breakout
        elif daily_bearish and st_short and donch_breakout_short:
            new_signal = -SIZE
        # Trigger 4: CRSI overbought + Supertrend short + volume (mean reversion in downtrend)
        elif crsi_overbought and st_short and vol_confirm:
            new_signal = -SIZE
        # Trigger 5: RSI bearish + Supertrend short + HMA aligned
        elif rsi_bearish and st_short and hma_trend_short:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 01:18
