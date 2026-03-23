# Strategy: mtf_4h_hma_rsi_donchian_12h_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.358 | -3.6% | -30.0% | 230 | FAIL |
| ETHUSDT | -0.361 | -11.5% | -32.8% | 246 | FAIL |
| SOLUSDT | 0.712 | +134.7% | -23.6% | 249 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.406 | +14.0% | -11.2% | 86 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #624: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: Building on proven 4h patterns (HMA+RSI+ATR showed Sharpe +0.879 on SOL),
this strategy combines 12h HMA trend filter with 4h RSI pullback entries and Donchian
breakout confirmation. Simpler than recent failed experiments (#612-#623) which had
too many regime filters causing 0 trades.

Key insights from 552 failed strategies:
1. Over-engineered regime switching = 0 trades (see #615, #619, #620, #621)
2. 4h timeframe works well with 12h HTF (not 1w which is too slow)
3. HMA is faster than EMA/KAMA for trend detection (less lag)
4. RSI pullback entries in trend direction have high win rate
5. Donchian breakout confirms trend strength before entry
6. Conservative sizing (0.28) + ATR stop controls drawdown

Why this might beat Sharpe=0.520:
- 12h HMA slope filter keeps us on right side of major moves (simpler than 1w)
- RSI pullback (40-60 range) enters on dips in uptrend, rallies in downtrend
- Donchian(20) breakout confirmation ensures momentum exists
- 2.5*ATR trailing stop limits losses on reversals
- Fewer filters = more trades (target 30-50/year on 4h)
- Discrete sizing (0.28) minimizes fee churn

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 30-50 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_12h_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    Upper = highest high over period
    Lower = lowest low over period
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=9)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H HMA FAST/SLOW CROSSOVER ===
        hma_cross_bull = hma_4h_fast[i] > hma_4h[i]
        hma_cross_bear = hma_4h_fast[i] < hma_4h[i]
        
        # === 4H HMA SLOPE (2 bars) ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-2] if i >= 2 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === RSI PULLBACK ZONES ===
        rsi_pullback_long = 40.0 <= rsi_14[i] <= 60.0
        rsi_pullback_short = 40.0 <= rsi_14[i] <= 60.0
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 12h bull trend + 4h pullback + Donchian confirmation ---
        # Condition 1: 12h HMA sloping up + price above 12h HMA
        # Condition 2: 4h HMA fast > slow (momentum)
        # Condition 3: RSI in pullback zone (40-60) or oversold (<45)
        # Condition 4: Price near or breaking Donchian upper (momentum)
        if hma_12h_slope_bull and price_above_hma_12h:
            if hma_cross_bull and hma_4h_slope_bull:
                if rsi_oversold or (rsi_pullback_long and close[i] > donchian_mid[i]):
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 12h bear trend + 4h bounce + Donchian confirmation ---
        # Condition 1: 12h HMA sloping down + price below 12h HMA
        # Condition 2: 4h HMA fast < slow (momentum)
        # Condition 3: RSI in pullback zone (40-60) or overbought (>55)
        # Condition 4: Price near or breaking Donchian lower (momentum)
        elif hma_12h_slope_bear and price_below_hma_12h:
            if hma_cross_bear and hma_4h_slope_bear:
                if rsi_overbought or (rsi_pullback_short and close[i] < donchian_mid[i]):
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 06:58
