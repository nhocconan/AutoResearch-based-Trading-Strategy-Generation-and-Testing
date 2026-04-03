# Strategy: mtf_4h_trailing_channel_chop_vol_12h_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.466 | +33.5% | -8.3% | 147 | PASS |
| ETHUSDT | 0.350 | +33.7% | -7.1% | 152 | PASS |
| SOLUSDT | -0.003 | +18.2% | -26.4% | 135 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.660 | +2.4% | -4.4% | 56 | FAIL |
| ETHUSDT | 0.902 | +15.3% | -5.0% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #006: 4h ATR Volatility Expansion + Choppiness + Trailing Channel

HYPOTHESIS: Combine the BEST elements from DB winners:
1. Choppiness regime filter (key meta-filter)
2. ATR-based trailing stop (proven 4h winners use this)
3. Trailing Donchian channel (not just breakout, but channel TRAILING)
4. Stricter volume confirmation (2.0x vs 1.5x)
5. Slightly tighter entry: CHOP < 45 (not < 50)

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: CHOP < 45 + price above trailing channel + HTF up = strong longs
- Bear: CHOP < 45 + price below trailing channel + HTF down = strong shorts
- Range: CHOP > 61.8 = SKIP (avoids whipsaws, the #1 killer)
- ATR trailing stop scales with volatility (handles 2022 crash)

KEY DIFFERENCE FROM #003:
- #003 had CHOP < 50 + 1.5x vol + basic Donchian = 306 trades
- #006: CHOP < 45 + 2.0x vol + TRAILING channel = ~150-200 trades
- Fewer, higher-quality signals = less fee drag = better Sharpe

TARGET: 100-200 total trades over 4 years (25-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trailing_channel_chop_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 45 = trending - GOOD to enter (stricter than usual 50)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_trailing_channel(high, low, period=20):
    """
    Trailing Donchian Channel - tracks highest high and lowest low
    Uses the channel BOTTOM for longs (support), TOP for shorts (resistance)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    channel_up, channel_lo = calculate_trailing_channel(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA) - stricter 2.0x threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # 200 for channel + 14 for CHOP + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(channel_up[i]) or np.isnan(channel_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER (stricter: <45, not <50) ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 45  # Stricter than usual 50
        
        # === HTF TREND: 12h EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION (2.0x, stricter than 1.5x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === TRAILING CHANNEL BREAKOUT ===
        # Long: price breaks ABOVE previous channel high
        # Short: price breaks BELOW previous channel low
        prev_channel_up = channel_up[i - 1]
        prev_channel_lo = channel_lo[i - 1]
        
        breakout_up = close[i] > prev_channel_up
        breakout_down = close[i] < prev_channel_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + breakout up + HTF trend up + volume spike ===
            if breakout_up and htf_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Trending + breakout down + HTF trend down + volume spike ===
            if breakout_down and htf_trend_down and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 07:29
