# Strategy: mtf_12h_kama_1d_hma_macd_adx_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.104 | +13.8% | -10.7% | 205 | FAIL |
| ETHUSDT | 0.103 | +24.2% | -18.4% | 217 | PASS |
| SOLUSDT | 0.578 | +87.7% | -24.2% | 220 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.518 | -4.2% | -11.0% | 78 | FAIL |
| SOLUSDT | 0.054 | +5.7% | -10.2% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #173: 12h KAMA Adaptive Trend + 1d HMA Filter + MACD Momentum + ADX

Hypothesis: 12h timeframe captures major crypto trends with fewer whipsaws than 4h.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends, slow in chop.
Combined with 1d HMA for higher timeframe bias, MACD histogram for momentum confirmation,
and ADX for trend strength filtering. This should outperform simple EMA/HMA strategies.

Why 12h KAMA might work:
- 12h bars filter intraday noise while catching multi-day trends
- KAMA adapts to market regime (faster in trends, slower in ranges)
- 1d HMA provides stable trend bias (avoids counter-trend trades)
- MACD histogram confirms momentum direction (avoids false breakouts)
- ADX > 18 filters chop while allowing sufficient trade count
- Conservative sizing (0.25) protects against 2022-style crashes

Learning from failures:
- #161 (12h Donchian): Sharpe=-0.334 - breakout alone fails without momentum filter
- #167 (12h Supertrend): Sharpe=-0.643 - supertrend whipsaws in ranges
- #166 (4h CRSI mean rev): Sharpe=-48.7 - mean reversion catastrophic on crypto
- #172 (4h MACD hist): Sharpe=-0.108 - close to breakeven, needs better HTF filter
- Trend following > mean reversion on crypto (BTC/ETH especially)
- Need MULTI-CONFIRMATION: trend + momentum + strength filters

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (conservative for drawdown control)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_macd_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - moves fast in trends, slow in chop.
    
    Efficiency Ratio (ER) = |Net Change| / Sum of Absolute Changes
    Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close - np.roll(close, er_period))
    net_change[:er_period] = np.abs(close[:er_period] - close[0])
    
    sum_abs_change = np.zeros(n)
    for i in range(er_period, n):
        sum_abs_change[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    sum_abs_change[:er_period] = sum_abs_change[er_period]
    
    # Avoid division by zero
    sum_abs_change = np.where(sum_abs_change == 0, 1e-10, sum_abs_change)
    er = net_change / sum_abs_change
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_macd_histogram(close, fast=12, slow=26, signal=9):
    """Calculate MACD Histogram (MACD line - Signal line)."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]), 
                        abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop[:period] = chop[period] if period < n else 50
    return chop

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    macd_hist = calculate_macd_histogram(close, 12, 26, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 = trending market (momentum more likely to continue)
        # Using 18 instead of 20/25 to ensure enough trades on 12h
        trend_strength = adx[i] > 18
        
        # === CHOPPINESS FILTER ===
        # CHOP < 50 = more trending than ranging (avoid extreme chop)
        not_choppy = chop[i] < 55
        
        # === KAMA ADAPTIVE TREND ===
        # Price above KAMA = bullish adaptive trend
        # Price below KAMA = bearish adaptive trend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === MACD HISTOGRAM MOMENTUM ===
        # Histogram positive = bullish momentum
        # Histogram negative = bearish momentum
        hist_positive = macd_hist[i] > 0
        hist_negative = macd_hist[i] < 0
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1d bullish + ADX trending + not choppy + price>KAMA + MACD hist positive
        if bull_trend_1d and trend_strength and not_choppy and price_above_kama and hist_positive:
            new_signal = SIZE_BASE
        
        # Short: 1d bearish + ADX trending + not choppy + price<KAMA + MACD hist negative
        if bear_trend_1d and trend_strength and not_choppy and price_below_kama and hist_negative:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 12:58
