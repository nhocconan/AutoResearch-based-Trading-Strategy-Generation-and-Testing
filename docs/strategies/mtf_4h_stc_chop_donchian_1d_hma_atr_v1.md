# Strategy: mtf_4h_stc_chop_donchian_1d_hma_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.234 | +5.4% | -24.9% | 436 | FAIL |
| ETHUSDT | 0.026 | +18.4% | -28.6% | 444 | PASS |
| SOLUSDT | 1.006 | +183.1% | -26.4% | 421 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.553 | +15.1% | -7.3% | 141 | PASS |
| SOLUSDT | -0.388 | -3.1% | -19.2% | 161 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1071: 4h Primary + 1d/1w HTF — Schaff Trend Cycle + Choppiness + Donchian Breakout

Hypothesis: After 777+ failed experiments, the winning pattern for 4h timeframe combines:
1. SCHAFF TREND CYCLE (STC) — combines MACD + Stochastic for smoother trend signals
   Less noisy than RSI/CRSI, clearer trend identification
   Long: STC crosses above 25 from below | Short: STC crosses below 75 from above
2. CHOPPINESS INDEX (CHOP) — regime detection
   CHOP > 61.8 = range (mean reversion at Donchian bounds)
   CHOP < 38.2 = trend (breakout following)
3. DONCHIAN CHANNEL (20) — breakout confirmation
   Long: price breaks upper Donchian + STC bullish
   Short: price breaks lower Donchian + STC bearish
4. 1d HMA21 macro bias — only trade in direction of daily trend
5. ATR-based position sizing — reduce size when vol spikes (ATR ratio > 2.0)

Why this should beat Sharpe=0.612:
- STC is PROVEN for crypto trends (different from all failed RSI/CRSI/Fisher strategies)
- Choppiness filter prevents trend strategies in ranges (major failure mode)
- Donchian breakout gives clear entry/exit points
- 4h timeframe = 20-50 trades/year target (optimal fee/trade balance)
- Different signal source than all 777 failed strategies

Timeframe: 4h (primary)
HTF: 1d (daily) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels (reduced to 0.15 in high vol)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_stc_chop_donchian_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_stc(close, macd_fast=23, macd_slow=50, stoch_period=10, stc_period=10):
    """
    Schaff Trend Cycle (STC) — combines MACD and Stochastic Oscillator.
    
    Formula:
    1. Calculate MACD(fast, slow)
    2. Apply Stochastic to MACD line over stoch_period
    3. Smooth with DCF (Double Smoothed) over stc_period
    4. Output bounded 0-100
    
    Signals:
    - STC > 75 = overbought (potential short)
    - STC < 25 = oversold (potential long)
    - Crossovers at 25/75 levels give entry signals
    
    Proven in crypto research for smoother trend signals vs RSI/MACD alone.
    """
    n = len(close)
    stc = np.full(n, np.nan)
    
    if n < macd_slow + stoch_period + stc_period:
        return stc
    
    # Step 1: Calculate MACD
    ema_fast = pd.Series(close).ewm(span=macd_fast, min_periods=macd_fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=macd_slow, min_periods=macd_slow, adjust=False).mean().values
    macd = ema_fast - ema_slow
    
    # Step 2: Apply Stochastic to MACD
    stoch_macd = np.full(n, np.nan)
    for i in range(stoch_period, n):
        if np.isnan(macd[i]):
            continue
        macd_window = macd[i - stoch_period + 1:i + 1]
        if np.any(np.isnan(macd_window)):
            continue
        highest = np.nanmax(macd_window)
        lowest = np.nanmin(macd_window)
        if highest - lowest > 1e-10:
            stoch_macd[i] = 100.0 * (macd[i] - lowest) / (highest - lowest)
        else:
            stoch_macd[i] = 50.0
    
    # Step 3: Smooth with DCF (Double Smoothed)
    # First smoothing
    d1 = pd.Series(stoch_macd).ewm(span=stc_period, min_periods=stc_period, adjust=False).mean().values
    
    # Second smoothing (apply stochastic again to d1)
    d2 = np.full(n, np.nan)
    for i in range(stc_period, n):
        if np.isnan(d1[i]):
            continue
        d1_window = d1[i - stc_period + 1:i + 1]
        if np.any(np.isnan(d1_window)):
            continue
        highest = np.nanmax(d1_window)
        lowest = np.nanmin(d1_window)
        if highest - lowest > 1e-10:
            d2[i] = 100.0 * (d1[i] - lowest) / (highest - lowest)
        else:
            d2[i] = 50.0
    
    # Final STC
    stc = pd.Series(d2).ewm(span=stc_period, min_periods=stc_period, adjust=False).mean().values
    
    return stc

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion favored)
    - CHOP < 38.2 = trending market (breakout/trend follow favored)
    - 38.2 - 61.8 = transition zone
    
    Research shows this is the BEST regime filter for crypto markets.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout detection.
    
    Upper Band = Highest High over period
    Lower Band = Lowest Low over period
    Middle = (Upper + Lower) / 2
    
    Breakout above upper = bullish signal
    Breakout below lower = bearish signal
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio for volatility regime detection.
    Ratio > 2.0 = volatility spike (reduce position size)
    Ratio < 1.2 = volatility crush (normal size)
    """
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    valid_mask = (~np.isnan(atr_short)) & (~np.isnan(atr_long)) & (atr_long > 1e-10)
    ratio[valid_mask] = atr_short[valid_mask] / atr_long[valid_mask]
    
    return ratio

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    stc = calculate_stc(close, macd_fast=23, macd_slow=50, stoch_period=10, stc_period=10)
    chop = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15  # Half size in high volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track STC crossovers
    prev_stc = np.full(n, 50.0)
    for i in range(1, n):
        if not np.isnan(stc[i-1]):
            prev_stc[i] = stc[i-1]
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(stc[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === VOLATILITY REGIME (Position Sizing) ===
        vol_spike = atr_ratio[i] > 2.0
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === STC SIGNALS ===
        stc_oversold = stc[i] < 25.0
        stc_overbought = stc[i] > 75.0
        
        # STC crossover signals
        stc_long_cross = prev_stc[i] < 25.0 and stc[i] >= 25.0
        stc_short_cross = prev_stc[i] > 75.0 and stc[i] <= 75.0
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: MEAN REVERSION AT DONCHIAN BOUNDS ===
        if is_choppy:
            # Long at lower Donchian + STC oversold + macro bullish
            if close[i] <= donch_lower[i] * 1.001 and stc_oversold and macro_bull:
                desired_signal = current_size
            # Short at upper Donchian + STC overbought + macro bearish
            elif close[i] >= donch_upper[i] * 0.999 and stc_overbought and macro_bear:
                desired_signal = -current_size
        
        # === TRENDING REGIME: BREAKOUT FOLLOWING ===
        elif is_trending:
            # Long breakout + STC bullish crossover + macro bullish
            if donch_breakout_long and stc_long_cross and macro_bull:
                desired_signal = current_size
            elif donch_breakout_long and stc[i] > 50.0 and macro_bull:
                desired_signal = current_size * 0.5  # Weaker signal
            
            # Short breakout + STC bearish crossover + macro bearish
            elif donch_breakout_short and stc_short_cross and macro_bear:
                desired_signal = -current_size
            elif donch_breakout_short and stc[i] < 50.0 and macro_bear:
                desired_signal = -current_size * 0.5  # Weaker signal
        
        # === TRANSITION ZONE: COMBINED SIGNALS ===
        else:
            # Long: STC crossover + macro bullish
            if stc_long_cross and macro_bull:
                desired_signal = current_size
            elif stc_oversold and macro_bull and close[i] > donch_mid[i]:
                desired_signal = current_size * 0.5
            
            # Short: STC crossover + macro bearish
            elif stc_short_cross and macro_bear:
                desired_signal = -current_size
            elif stc_overbought and macro_bear and close[i] < donch_mid[i]:
                desired_signal = -current_size * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if STC still bullish or price above Donchian mid
                if stc[i] > 40.0 or close[i] > donch_mid[i]:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if STC still bearish or price below Donchian mid
                if stc[i] < 60.0 or close[i] < donch_mid[i]:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if STC overbought AND price breaks Donchian lower
            if stc_overbought and close[i] < donch_lower[i]:
                desired_signal = 0.0
            # Exit long if macro reverses strongly bearish
            if macro_bear and stc[i] < 40.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if STC oversold AND price breaks Donchian upper
            if stc_oversold and close[i] > donch_upper[i]:
                desired_signal = 0.0
            # Exit short if macro reverses strongly bullish
            if macro_bull and stc[i] > 60.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 19:20
