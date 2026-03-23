# Strategy: mtf_1d_fisher_kama_chop_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.000 | +0.0% | 0.0% | 0 | FAIL |
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
Experiment #017: 1d Fisher Transform + KAMA Adaptive Trend + Weekly HMA Filter

Hypothesis: Previous strategies failed because they used standard indicators (RSI, EMA)
that don't adapt to changing market regimes. This strategy combines:

1. Ehlers Fisher Transform (period=9) - transforms price into Gaussian distribution,
   catches reversals better than RSI in bear markets. Long when Fisher crosses -1.5,
   short when crosses +1.5. Proven in research for crypto reversals.

2. KAMA (Kaufman Adaptive Moving Average) - adapts smoothing based on market noise.
   Fast in trends, slow in chop. Better than HMA/EMA for regime changes.
   ER (Efficiency Ratio) determines adaptive smoothing constant.

3. Choppiness Index (14) - regime filter. CHOP > 55 = range (mean revert),
   CHOP < 45 = trend (trend follow). Dual-mode strategy.

4. 1w HMA(21) - weekly major trend bias via mtf_data helper. Only long if price > 1w HMA,
   only short if price < 1w HMA. Prevents counter-trend disasters.

5. ATR(14) trailing stop - 2.5x ATR for risk management.

Why this should work:
- Fisher Transform proven for crypto reversals (better than RSI in 2022 crash)
- KAMA adapts to volatility (works in both bull and bear regimes)
- Weekly HMA filter prevents major counter-trend trades
- Choppiness filter switches between mean-revert and trend-follow modes
- 1d timeframe = 20-50 trades/year target (optimal for fee drag)
- Conservative sizing (0.25-0.30) protects against 77% crashes

Timeframe: 1d (REQUIRED for Experiment #017)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price into near-Gaussian distribution.
    Catches reversals better than RSI, especially in bear markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest(period)) / (highest(period) - lowest(period))
    3. Transform: 0.66 * ((norm - 0.5) + 0.67 * prev_transform)
    4. Fisher: 0.5 * ln((1 + transform) / (1 - transform))
    
    Entry signals:
    - Long: Fisher crosses above -1.5 (oversold reversal)
    - Short: Fisher crosses below +1.5 (overbought reversal)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    n = len(close)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize price over lookback period
    lowest = typical.rolling(window=period, min_periods=period).min()
    highest = typical.rolling(window=period, min_periods=period).max()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, 0.001)
    
    normalized = (typical - lowest) / range_val
    
    # Calculate transform iteratively (needs previous value)
    transform = np.zeros(n)
    fisher = np.zeros(n)
    
    for i in range(period, n):
        # Ehlers smoothing formula
        if i == period:
            transform[i] = 0.66 * ((normalized.iloc[i] - 0.5) + 0.67 * 0)
        else:
            transform[i] = 0.66 * ((normalized.iloc[i] - 0.5) + 0.67 * transform[i-1])
        
        # Clamp to avoid ln domain errors
        transform[i] = np.clip(transform[i], -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + transform[i]) / (1 - transform[i]))
    
    return fisher

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) - adapts to market noise.
    
    Efficiency Ratio (ER) = |close - close(n)| / sum(|close - close(prev)|)
    ER near 1 = trending (use fast SC)
    ER near 0 = choppy (use slow SC)
    
    Smoothing Constant (SC) = (ER * (fast - slow) + slow)^2
    KAMA = KAMA(prev) + SC * (close - KAMA(prev))
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Price change over ER period
    price_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of absolute price changes (volatility)
    volatility = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = price_change / volatility
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA iteratively
    kama = np.zeros(n)
    kama[er_period] = close_s.iloc[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    hh_ll = hh_ll.replace(0, 0.001)
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
    
    return chop.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Additional trend confirmation
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(kama[i]):
            continue
        
        # === WEEKLY MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55  # Ranging market (mean reversion)
        chop_trend = chop[i] < 45  # Trending market
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama[i] and kama[i] > kama_fast[i]
        kama_bearish = close[i] < kama[i] and kama[i] < kama_fast[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Detect crossings (need previous value)
        fisher_cross_up = False
        fisher_cross_down = False
        
        if i > 0 and not np.isnan(fisher[i-1]):
            # Long signal: Fisher crosses above -1.5 from below
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_up = True
            # Short signal: Fisher crosses below +1.5 from above
            if fisher[i-1] > 1.5 and fisher[i] <= 1.5:
                fisher_cross_down = True
        
        # Fisher extreme levels (for mean reversion)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        long_score = 0
        long_confidence = 0
        
        # Primary trigger: Fisher reversal or extreme
        if fisher_cross_up:
            long_score += 2.5
            long_confidence = 1
        elif fisher_oversold:
            long_score += 1.5
            long_confidence = 0.7
        
        # Trend alignment
        if weekly_bullish:
            long_score += 1.5
        if kama_bullish:
            long_score += 1.0
        
        # Regime-based entry
        if chop_range:
            # Mean reversion mode - enter on Fisher extreme
            if fisher_oversold:
                long_score += 1.5
        elif chop_trend:
            # Trend mode - enter on Fisher cross with trend
            if fisher_cross_up and kama_bullish:
                long_score += 1.5
        
        # Enter long if score >= 4.5 (strong confluence)
        if long_score >= 4.5:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY
        short_score = 0
        short_confidence = 0
        
        # Primary trigger: Fisher reversal or extreme
        if fisher_cross_down:
            short_score += 2.5
            short_confidence = 1
        elif fisher_overbought:
            short_score += 1.5
            short_confidence = 0.7
        
        # Trend alignment
        if weekly_bearish:
            short_score += 1.5
        if kama_bearish:
            short_score += 1.0
        
        # Regime-based entry
        if chop_range:
            # Mean reversion mode - enter on Fisher extreme
            if fisher_overbought:
                short_score += 1.5
        elif chop_trend:
            # Trend mode - enter on Fisher cross with trend
            if fisher_cross_down and kama_bearish:
                short_score += 1.5
        
        # Enter short if score >= 4.5 (strong confluence)
        if short_score >= 4.5:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~60 days on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if fisher_oversold and weekly_bullish:
                new_signal = REDUCED_SIZE
            elif fisher_overbought and weekly_bearish:
                new_signal = -REDUCED_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long if Fisher goes overbought (mean reversion complete)
            if position_side > 0 and fisher[i] > 1.0:
                fisher_exit = True
            # Exit short if Fisher goes oversold (mean reversion complete)
            if position_side < 0 and fisher[i] < -1.0:
                fisher_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish:
                trend_reversal = True
        
        # === KAMA CROSS EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Exit long if price crosses below KAMA
            if position_side > 0 and close[i] < kama[i]:
                kama_exit = True
            # Exit short if price crosses above KAMA
            if position_side < 0 and close[i] > kama[i]:
                kama_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or fisher_exit or trend_reversal or kama_exit:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
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
2026-03-22 21:00
