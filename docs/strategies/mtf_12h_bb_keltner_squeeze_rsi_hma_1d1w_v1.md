# Strategy: mtf_12h_bb_keltner_squeeze_rsi_hma_1d1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.915 | +0.4% | -9.4% | 102 | FAIL |
| ETHUSDT | 0.122 | +25.2% | -11.2% | 105 | PASS |
| SOLUSDT | 0.129 | +26.2% | -14.3% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.686 | -0.1% | -6.5% | 33 | FAIL |
| SOLUSDT | 0.482 | +11.1% | -5.3% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #692: 12h Primary + 1d/1w HTF — BB/Keltner Squeeze + RSI + HMA Trend

Hypothesis: Volatility compression (BB inside Keltner) precedes major breakouts.
Combined with RSI extremes and HTF trend bias, this captures both mean-reversion
in ranges and breakout momentum in trends. 12h TF reduces noise vs 4h while
maintaining sufficient trade frequency (target 30-50 trades/year).

Key Differences from #691:
1. BB/Keltner Squeeze detection instead of Choppiness (more reliable for breakouts)
2. RSI(14) with SMA(200) filter instead of CRSI (simpler, more trades)
3. 1w HMA as additional HTF filter (stronger trend bias than just 1d)
4. Looser RSI thresholds (25/75 not 15/85) to ensure trade frequency
5. ATR-based position sizing adjustment (reduce size in high vol)

Why this should work:
- BB/Keltner squeeze is proven in traditional markets (John Carter, Squeeze Pro)
- 12h TF worked in #682 (Sharpe=0.404) and #686 (Sharpe=0.285)
- SMA200 filter prevents counter-trend trades that failed in #685/#690
- Multiple HTF filters (1d + 1w HMA) reduce whipsaw

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_keltner_squeeze_rsi_hma_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - volatility bands around SMA."""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / (sma + 1e-10) * 100
    return upper, lower, sma, bandwidth

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=1.5):
    """Keltner Channels - ATR-based volatility channels."""
    n = len(close)
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return upper, lower, atr

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Shift to align with price (ewm adds one NaN at start)
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    kc_upper, kc_lower, atr_12h = calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=1.5)
    rsi_12h = calculate_rsi(close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(adx_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === VOLATILITY REGIME (BB inside Keltner = Squeeze) ===
        is_squeeze = (bb_upper[i] <= kc_upper[i]) and (bb_lower[i] >= kc_lower[i])
        is_expansion = (bb_upper[i] > kc_upper[i]) or (bb_lower[i] < kc_lower[i])
        
        # === TREND BIAS (HTF HMA) ===
        # Both 1d and 1w must agree for strong signal
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        trend_bullish_1w = close[i] > hma_1w_aligned[i]
        trend_bearish_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias when both HTF agree
        trend_strong_bullish = trend_bullish_1d and trend_bullish_1w
        trend_strong_bearish = trend_bearish_1d and trend_bearish_1w
        trend_neutral = not trend_strong_bullish and not trend_strong_bearish
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ADX STRENGTH ===
        adx_strong = adx_12h[i] > 25
        adx_weak = adx_12h[i] < 20
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in high volatility (ATR expansion)
        atr_ratio = atr_12h[i] / (np.nanmedian(atr_12h[max(0,i-100):i+1]) + 1e-10)
        if atr_ratio > 1.5:
            current_size = REDUCED_SIZE
        
        # === MEAN REVERSION MODE (Squeeze/Range) ===
        if is_squeeze or adx_weak:
            # Long: RSI oversold + above SMA200 + bullish HTF bias
            if rsi_12h[i] < 30 and above_sma200 and (trend_bullish_1d or trend_neutral):
                desired_signal = current_size
            
            # Short: RSI overbought + below SMA200 + bearish HTF bias
            elif rsi_12h[i] > 70 and below_sma200 and (trend_bearish_1d or trend_neutral):
                desired_signal = -current_size
            
            # Weaker signals without SMA200 confirmation
            elif rsi_12h[i] < 25 and trend_bullish_1d:
                desired_signal = current_size * 0.5
            elif rsi_12h[i] > 75 and trend_bearish_1d:
                desired_signal = -current_size * 0.5
        
        # === TREND FOLLOWING MODE (Expansion/Strong ADX) ===
        elif is_expansion and adx_strong:
            # Long breakout: price above BB upper + strong bullish trend
            if close[i] > bb_upper[i] and trend_strong_bullish and above_sma200:
                desired_signal = current_size
            
            # Short breakout: price below BB lower + strong bearish trend
            elif close[i] < bb_lower[i] and trend_strong_bearish and below_sma200:
                desired_signal = -current_size
            
            # Weaker breakout with single HTF confirmation
            elif close[i] > bb_upper[i] and trend_bullish_1d and above_sma200:
                desired_signal = current_size * 0.5
            elif close[i] < bb_lower[i] and trend_bearish_1d and below_sma200:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if RSI not overbought and trend still intact
                if rsi_12h[i] < 75 and trend_bullish_1d:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if RSI not oversold and trend still intact
                if rsi_12h[i] > 25 and trend_bearish_1d:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI overbought OR trend reverses below both HTF HMA
        if in_position and position_side > 0:
            if rsi_12h[i] > 80:
                desired_signal = 0.0
            elif close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]:
                desired_signal = 0.0
        
        # Short exit: RSI oversold OR trend reverses above both HTF HMA
        if in_position and position_side < 0:
            if rsi_12h[i] < 20:
                desired_signal = 0.0
            elif close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
2026-03-23 12:48
