# Strategy: mtf_12h_donchian_1d_1w_hma_adx_vol_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.046 | +22.5% | -10.1% | 169 | PASS |
| ETHUSDT | -0.473 | +1.2% | -14.3% | 190 | FAIL |
| SOLUSDT | 0.839 | +108.1% | -10.8% | 183 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.322 | -5.5% | -10.4% | 63 | FAIL |
| SOLUSDT | 0.395 | +11.0% | -7.1% | 63 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian Breakout + 1d/1w HMA Trend Filter + ADX + Volume

Hypothesis: After 22 experiments, the clearest pattern is:
1. Lower TFs (15m-1h) suffer from noise and fee drag - most have negative Sharpe
2. 12h timeframe is UNDERUTILIZED - only 2 attempts (#011, #017), #017 had Sharpe=0.109
3. Donchian breakouts work BEST on higher timeframes (classic trend following)
4. 1d/1w HMA provides robust trend bias without overfitting
5. ADX > 18 filters out choppy ranges where breakouts fail
6. Volume confirmation is CRITICAL - missing from most failed strategies

This 12h strategy combines:

1. Donchian Channel (20): Breakout above 20-bar high = long, below 20-bar low = short.
   Classic trend-following entry, proven on higher timeframes.

2. 1d HMA + 1w HMA dual trend filter: At least one must agree for direction.
   More flexible than requiring both = more trades while maintaining edge.

3. ADX(14) > 18: Confirms trending conditions, avoids breakout failures in ranges.

4. Volume Spike: Volume > 1.3x 20-bar average confirms breakout conviction.
   Lower threshold = more trades while still filtering fakeouts.

5. Asymmetric Sizing: 0.35 for strong signals (all filters agree), 0.25 for moderate.
   Conservative sizing protects from 2022-style crashes.

6. ATR Trailing Stop: 3.0*ATR protects from reversals (wider for 12h TF).

Why this should beat current best (Sharpe=0.123):
- 12h TF naturally filters noise - fewer but higher quality trades
- Donchian breakouts are PROVEN trend-following (Turtle Trading legacy)
- Dual HTF filter (1d+1w) more robust than single 4h/12h filter
- Volume confirmation addresses the #1 failure mode (fake breakouts)
- Conservative sizing (0.25-0.35) limits drawdown in bear markets

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.0 * ATR(14) trailing (wider for 12h)
Target trades: 20-50/year on 12h (optimal per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_1w_hma_adx_vol_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    
    return upper.values, lower.values, middle.values

def calculate_volume_spike(volume, lookback=20, threshold=1.3):
    """Detect volume spikes above threshold * average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume > (threshold * vol_avg)
    return vol_spike

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    vol_spike = calculate_volume_spike(volume, lookback=20, threshold=1.3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.35  # All filters agree
    SIZE_MODERATE = 0.25  # Partial confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA - at least one must agree) ===
        # More flexible: price above at least one HMA for bull, below at least one for bear
        price_vs_1d = close[i] - hma_1d_aligned[i]
        price_vs_1w = close[i] - hma_1w_aligned[i]
        
        bull_htf = (price_vs_1d > 0) or (price_vs_1w > 0)  # At least one bullish
        bear_htf = (price_vs_1d < 0) or (price_vs_1w < 0)  # At least one bearish
        
        # Strong HTF: both agree
        bull_htf_strong = (price_vs_1d > 0) and (price_vs_1w > 0)
        bear_htf_strong = (price_vs_1d < 0) and (price_vs_1w < 0)
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 18  # Trending market (slightly lower threshold)
        
        # === DONCHIAN BREAKOUT ===
        # Use previous bar's Donchian levels to avoid look-ahead
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0  # Count confirming filters
        
        # LONG ENTRY: HTF bull + Donchian breakout + volume/ADX/DI confirmation
        if bull_htf and donchian_breakout_long:
            signal_strength += 2  # HTF trend + Donchian breakout (core signals)
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            if di_bull:
                signal_strength += 1  # DI direction
            
            if bull_htf_strong:
                signal_strength += 1  # Both HTF agree
            
            # Assign size based on confirmation count
            if signal_strength >= 5:
                new_signal = SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: HTF bear + Donchian breakout + volume/ADX/DI confirmation
        elif bear_htf and donchian_breakout_short:
            signal_strength += 2  # HTF trend + Donchian breakout (core signals)
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            if di_bear:
                signal_strength += 1  # DI direction
            
            if bear_htf_strong:
                signal_strength += 1  # Both HTF agree
            
            # Assign size based on confirmation count
            if signal_strength >= 5:
                new_signal = -SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend strongly reverses against position
            if position_side > 0 and bear_htf_strong:
                trend_exit = True
            if position_side < 0 and bull_htf_strong:
                trend_exit = True
            
            # Exit if Donchian channel middle crosses against position
            if position_side > 0 and close[i] < donchian_middle[i]:
                trend_exit = True
            if position_side < 0 and close[i] > donchian_middle[i]:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 20:11
