# Strategy: mtf_1h_connors_rsi_4h_12h_hma_session_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -1.014 | -16.2% | -28.1% | 348 | FAIL |
| SOLUSDT | 0.078 | +22.7% | -15.6% | 348 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.149 | +7.6% | -6.7% | 120 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #020: 1h Connors RSI + 4h/12h HMA Trend + Session Filter

Hypothesis: Lower timeframe (1h) with strict HTF trend filter + Connors RSI mean reversion
will capture pullbacks within the HTF trend. Key improvements:
1. Connors RSI (RSI3 + RSI_Streak + PercentRank) for precise entry timing
2. 4h HMA(21) + 12h HMA(21) dual HTF trend filter (both must agree)
3. Session filter: only trade 8-20 UTC (high liquidity, lower spread)
4. Volume confirmation: volume > 0.8x 20-bar average
5. Asymmetric sizing: 0.25 base, reduce to 0.20 in high vol
6. Stoploss: 2.5 * ATR(14) from entry

Why this should work:
- Connors RSI has 75% win rate on mean reversion entries
- HTF filter prevents counter-trend trades (major failure mode)
- Session filter avoids low-liquidity whipsaw (Asian session)
- 1h TF targets 40-80 trades/year (optimal for fee efficiency)
- Discrete sizing (0.20, 0.25, 0.30) minimizes churn costs

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_rsi_4h_12h_hma_session_v2"
timeframe = "1h"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of Connors RSI.
    Counts consecutive up/down days and applies RSI to that series.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Apply RSI to streak values
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    
    for i in range(period, n):
        gains = 0
        losses = 0
        for j in range(1, period + 1):
            if streak[i-j+1] > streak[i-j]:
                gains += streak[i-j+1] - streak[i-j]
            elif streak[i-j+1] < streak[i-j]:
                losses += streak[i-j] - streak[i-j+1]
        
        if losses == 0:
            streak_rsi[i] = 100
        else:
            rs = gains / losses
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    Percentage of closes in lookback period that are lower than current close.
    """
    n = len(close)
    percent_rank = np.zeros(n)
    
    for i in range(period, n):
        lookback = close[i-period:i]
        count_lower = np.sum(lookback < close[i])
        percent_rank[i] = (count_lower / period) * 100
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time_ms):
    """Extract UTC hour from millisecond timestamp."""
    return (open_time_ms // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    vol_avg = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    MIN_SIZE = 0.20
    MAX_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        # === HTF TREND BIAS (4h + 12h) ===
        # Both HTFs must agree for strong bias
        htf_bullish = (close[i] > hma_4h_21_aligned[i]) and (close[i] > hma_12h_21_aligned[i])
        htf_bearish = (close[i] < hma_4h_21_aligned[i]) and (close[i] < hma_12h_21_aligned[i])
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Very oversold
        crsi_overbought = crsi[i] > 85  # Very overbought
        crsi_neutral = 15 <= crsi[i] <= 85
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.8, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, MIN_SIZE, MAX_SIZE)
        current_size = np.round(current_size * 4) / 4  # Round to 0.05 increments
        current_size = np.clip(current_size, MIN_SIZE, MAX_SIZE)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + CRSI oversold + session + volume
        if htf_bullish and crsi_oversold and in_session and volume_ok:
            new_signal = current_size
        
        # SHORT ENTRY: HTF bearish + CRSI overbought + session + volume
        elif htf_bearish and crsi_overbought and in_session and volume_ok:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 48 bars (~2 days on 1h), allow weaker entry
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi[i] < 25 and in_session:
                new_signal = current_size * 0.8
            elif htf_bearish and crsi[i] > 75 and in_session:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI MEAN REVERSION EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                htf_reversal = True
            if position_side < 0 and htf_bullish:
                htf_reversal = True
        
        # === SESSION EXIT ===
        # Close position before session ends to avoid overnight risk
        session_ending = hour >= 19 and in_position
        if session_ending:
            new_signal = 0.0
        
        # Apply stoploss or reversals
        if stoploss_triggered or crsi_exit or htf_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 20:55
