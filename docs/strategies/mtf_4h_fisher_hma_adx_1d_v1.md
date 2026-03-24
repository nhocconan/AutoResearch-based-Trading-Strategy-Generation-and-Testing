# Strategy: mtf_4h_fisher_hma_adx_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.513 | -7.8% | -31.2% | 1030 | FAIL |
| ETHUSDT | -0.081 | +11.2% | -27.2% | 995 | FAIL |
| SOLUSDT | 0.600 | +92.2% | -20.9% | 1026 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.274 | +10.7% | -16.5% | 305 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #429: 4h Primary + 1d HTF — Fisher Transform + HMA Trend + ADX Filter

Hypothesis: 4h timeframe with daily bias should produce 20-50 trades/year with
better risk-adjusted returns than recent experiments. Key innovations:
1. Ehlers Fisher Transform for reversal entries — catches turns in bear/range markets
2. 1d HMA(21) for overall trend bias (bull/bear filter)
3. ADX(14) to distinguish trending vs ranging regimes
4. Simple entry conditions to ensure adequate trade frequency (avoid 0-trade failure)
5. ATR(14) trailing stoploss at 2.5x for risk management

Why this should beat #417 (Sharpe=0.042) and current best (Sharpe=0.612):
- Fisher Transform is more sensitive than RSI for reversal detection
- 4h has better trade frequency than 1d while maintaining quality
- ADX filter prevents entries in low-volatility chop
- Simpler logic than triple-regime approaches that generated 0 trades
- 1d HMA bias is stronger than 4h for overall direction

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_adx_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    Signals: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    for i in range(period, n):
        highest = close[i-period+1:i+1].max()
        lowest = close[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            fisher[i] = 0.0
            continue
        
        X = 0.67 * (close[i] - lowest) / (highest - lowest) - 0.33
        X = np.clip(X, -0.999, 0.999)  # Prevent log domain error
        
        fisher[i] = 0.5 * np.log((1.0 + X) / (1.0 - X + 1e-10))
    
    # Fisher signal line (1-period lag for crossover detection)
    for i in range(1, n):
        if not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_s.values
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(adx[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === ADX REGIME ===
        is_trending = adx[i] > 25.0  # ADX > 25 = trending
        is_ranging = adx[i] < 20.0   # ADX < 20 = ranging
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5  # Cross above -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5   # Cross below +1.5
        
        # Fisher extreme levels for mean reversion
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — Multiple confluence conditions (any one can trigger)
        long_bias = price_above_hma_1d or hma_bullish
        
        if long_bias:
            # Condition 1: Fisher reversal in trending market
            if is_trending and fisher_long:
                desired_signal = position_size
            # Condition 2: Fisher extreme in ranging market (mean reversion)
            elif is_ranging and fisher_oversold:
                desired_signal = position_size
            # Condition 3: HMA bullish + Fisher pullback
            elif hma_bullish and fisher[i] < 0.0:
                desired_signal = position_size * 0.7
            # Condition 4: Price above SMA200 + Fisher signal
            elif close[i] > sma_200[i] and fisher_long:
                desired_signal = position_size
        
        # SHORT SETUP — Multiple confluence conditions (any one can trigger)
        short_bias = price_below_hma_1d or hma_bearish
        
        if short_bias:
            # Condition 1: Fisher reversal in trending market
            if is_trending and fisher_short:
                desired_signal = -position_size
            # Condition 2: Fisher extreme in ranging market (mean reversion)
            elif is_ranging and fisher_overbought:
                desired_signal = -position_size
            # Condition 3: HMA bearish + Fisher rally
            elif hma_bearish and fisher[i] > 0.0:
                desired_signal = -position_size * 0.7
            # Condition 4: Price below SMA200 + Fisher signal
            elif close[i] < sma_200[i] and fisher_short:
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === FISHER EXTREME EXIT ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or hma_bullish):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_1d or hma_bearish):
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
2026-03-23 10:19
