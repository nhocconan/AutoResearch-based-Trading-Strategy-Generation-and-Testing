# Strategy: mtf_4h_chop_hma_rsi_donchian_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.375 | -1.2% | -22.7% | 337 | FAIL |
| ETHUSDT | -0.263 | -0.9% | -29.3% | 363 | FAIL |
| SOLUSDT | 0.581 | +91.3% | -28.3% | 384 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.071 | +5.8% | -17.7% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #399: 4h Primary + 1d HTF — Choppiness Regime + HMA + RSI + Donchian

Hypothesis: Combining Choppiness Index regime detection with HMA trend + RSI pullback
entries + Donchian breakout confirmation will beat Sharpe=0.612.

Key innovations vs current best (#394 Sharpe=0.301, best Sharpe=0.612):
1. Choppiness Index (CHOP) for regime: >61.8 = range (mean revert), <38.2 = trend
2. HMA(16/48) crossover for trend direction (proven in multiple winning strategies)
3. RSI(7) with DYNAMIC thresholds based on regime (not fixed 30/70)
4. Donchian(20) breakout for entry timing confirmation
5. 1d HTF HMA for overall bias filter
6. ATR(14) trailing stoploss (2.5x for longs, 2.0x for shorts)
7. Discrete position sizing: 0.0, ±0.28 to minimize fee churn

Why this should beat Sharpe=0.612:
- CHOP regime filter proven in #394 notes (ETH Sharpe +0.923 with CHOP+CRSI)
- HMA crossover proven in mtf_hma_rsi_zscore_v1 (Sharpe=5.4 baseline)
- 4h TF = target 25-45 trades/year = minimal fee drag (~1.5-2.5%)
- Different signal combination than #394 (adds RSI dynamic thresholds + Donchian)
- More trades than Fisher-based #396 (which only got Sharpe=0.017)

Target: Sharpe > 0.612, 25-50 trades/year, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_hma_rsi_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        # Calculate ATR sum over period
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (target 25-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        # Neutral zone: 38.2 <= CHOP <= 61.8
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI WITH DYNAMIC THRESHOLDS ===
        # In trending regime: use wider thresholds (25/75)
        # In choppy regime: use tighter thresholds (35/65)
        if is_trending:
            rsi_oversold = 25.0
            rsi_overbought = 75.0
        elif is_choppy:
            rsi_oversold = 35.0
            rsi_overbought = 65.0
        else:
            rsi_oversold = 30.0
            rsi_overbought = 70.0
        
        rsi_long = rsi_7[i] < rsi_oversold
        rsi_short = rsi_7[i] > rsi_overbought
        
        # === DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5  # Reduce to 50% in extreme vol
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7  # Reduce to 70%
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple confluence paths
        long_bias = price_above_hma_1d  # HTF bullish
        
        if long_bias:
            if is_trending and hma_bullish:
                # Trend following: HMA bullish + RSI not overbought
                if rsi_14[i] < rsi_overbought:
                    desired_signal = position_size
            elif is_choppy:
                # Mean reversion in range: RSI oversold + near Donchian lower
                if rsi_long and close[i] < (donchian_lower[i] + 0.02 * close[i]):
                    desired_signal = position_size
            elif hma_bullish and rsi_long:
                # Pullback in uptrend with RSI confirmation
                desired_signal = position_size
            elif donchian_long and hma_bullish:
                # Donchian breakout with trend confirmation
                desired_signal = position_size
        
        # SHORT SETUP - Multiple confluence paths
        short_bias = price_below_hma_1d  # HTF bearish
        
        if short_bias:
            if is_trending and hma_bearish:
                # Trend following: HMA bearish + RSI not oversold
                if rsi_14[i] > rsi_oversold:
                    desired_signal = -position_size
            elif is_choppy:
                # Mean reversion in range: RSI overbought + near Donchian upper
                if rsi_short and close[i] > (donchian_upper[i] - 0.02 * close[i]):
                    desired_signal = -position_size
            elif hma_bearish and rsi_short:
                # Rally in downtrend with RSI confirmation
                desired_signal = -position_size
            elif donchian_short and hma_bearish:
                # Donchian breakdown with trend confirmation
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (Asymmetric: tighter on shorts) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr  # 2.5x for longs
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr  # 2.0x for shorts
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXIT (extreme reached) ===
        if in_position and position_side > 0 and rsi_14[i] > rsi_overbought:
            # Long exit when RSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < rsi_oversold:
            # Short exit when RSI reaches oversold
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = position_size
            elif position_side < 0 and short_bias:
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
                # Position flip
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
2026-03-23 09:49
