# Strategy: mtf_4h_kama_trend_1d_hma_chop_adx_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.033 | -33.0% | -42.6% | 798 | FAIL |
| ETHUSDT | 0.025 | +16.3% | -27.6% | 824 | PASS |
| SOLUSDT | 0.530 | +90.0% | -27.7% | 801 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.258 | +10.1% | -14.9% | 255 | PASS |
| SOLUSDT | 0.755 | +23.8% | -19.3% | 271 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #262: 4h KAMA Trend with 1d HMA Bias and Choppiness Regime Filter

Hypothesis: After 225+ failed experiments, complexity is the enemy. This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than EMA/HMA
2. 1d HMA for directional bias (proven in successful strategies)
3. Choppiness Index to detect trend vs range regime
4. ADX filter for trend strength
5. Asymmetric sizing: larger in strong trends, smaller in weak trends
6. Simpler entry logic than #256 ensemble (which failed with Sharpe=-0.231)

Why this might work:
- KAMA adjusts efficiency ratio based on price movement - stays flat in ranges, follows in trends
- Choppiness Index > 61.8 = range (reduce size or stay flat), < 38.2 = trend (full size)
- 1d HMA bias prevents counter-trend trades that destroy Sharpe in 2022 crash
- ADX > 25 ensures we only trade when trend has momentum
- Fewer but higher quality trades = less fee drag

Key differences from #256 (failed ensemble):
- Single coherent signal instead of voting system
- KAMA instead of HMA crossover (more adaptive)
- Choppiness regime filter (missing from #256)
- Looser entry thresholds to ensure >=10 trades per symbol
- Asymmetric sizing based on regime strength

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels based on regime
Stoploss: 3.0 * ATR(14) trailing (wider to avoid whipsaws)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_trend_1d_hma_chop_adx_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - smooth in ranges, responsive in trends.
    Efficiency Ratio (ER) = |close - close_n| / sum(|close_i - close_i-1|)
    Smoothing Constant = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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
    n = len(close)
    
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
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = 0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    ema_50 = calculate_ema(close, 50)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels based on regime (Rule 4)
    SIZE_STRONG = 0.35  # Strong trend regime
    SIZE_MODERATE = 0.25  # Moderate trend
    SIZE_WEAK = 0.15  # Weak trend / transitional
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = intermediate trend bias (hard filter for direction)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        # Choppiness Index: > 61.8 = range, < 38.2 = trend
        is_trend_regime = chop[i] < 50  # Looser threshold to ensure trades
        is_range_regime = chop[i] >= 50
        
        # ADX trend strength
        is_strong_trend = adx[i] > 30
        is_moderate_trend = 20 < adx[i] <= 30
        is_weak_trend = adx[i] <= 20
        
        # === DIRECTIONAL SIGNALS ===
        # KAMA slope (trend direction)
        kama_slope_up = kama[i] > kama[i-3] if not np.isnan(kama[i-3]) else False
        kama_slope_down = kama[i] < kama[i-3] if not np.isnan(kama[i-3]) else False
        
        # Price vs KAMA (trend confirmation)
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Price vs EMA50 (medium-term trend)
        price_above_ema50 = close[i] > ema_50[i]
        price_below_ema50 = close[i] < ema_50[i]
        
        # DI crossover (momentum)
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        position_size = SIZE_MODERATE
        
        # Determine position size based on regime
        if is_strong_trend and is_trend_regime:
            position_size = SIZE_STRONG
        elif is_moderate_trend:
            position_size = SIZE_MODERATE
        else:
            position_size = SIZE_WEAK
        
        # LONG ENTRY: Need trend alignment across multiple timeframes
        # Must have: 1d bias up + KAMA up + price above KAMA + ADX confirms trend
        long_conditions = (
            bull_trend_1d and  # 1d HMA bias
            kama_slope_up and  # KAMA trending up
            price_above_kama and  # Price above KAMA
            di_bullish and  # DI confirms momentum
            (is_trend_regime or is_moderate_trend)  # Not in extreme range
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and  # 1d HMA bias
            kama_slope_down and  # KAMA trending down
            price_below_kama and  # Price below KAMA
            di_bearish and  # DI confirms momentum
            (is_trend_regime or is_moderate_trend)  # Not in extreme range
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.0 * ATR below highest close
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.0 * ATR above lowest close
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 14:37
