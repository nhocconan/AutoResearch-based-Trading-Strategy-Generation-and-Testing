# Strategy: mtf_4h_dual_regime_hma_rsi_bb_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.904 | -6.0% | -14.5% | 229 | FAIL |
| ETHUSDT | -0.419 | +4.6% | -12.0% | 206 | FAIL |
| SOLUSDT | 0.090 | +23.7% | -23.6% | 213 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.289 | +9.6% | -7.6% | 78 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #709: 4h Primary + 1d HTF — Dual Regime (Trend + Mean Reversion)

Hypothesis: Market alternates between trending and ranging regimes. Use ADX to detect
regime, then apply appropriate strategy:
- ADX > 25 (Trend): HMA trend + RSI pullback entries
- ADX < 20 (Range): Bollinger Band mean reversion

Key improvements over failed experiments:
1. Simpler entry logic (no CRSI/Choppiness combo that failed in #697-#708)
2. Dual regime adapts to market conditions (unlike pure trend strategies)
3. 1d HMA for strong trend bias (proven in current best Sharpe=0.612)
4. Looser RSI thresholds (30/70 not 25/75) to ensure trade frequency
5. ATR-based stoploss + take profit for risk management

Why this should work:
- 4h TF worked in current best (Sharpe=0.612)
- Dual regime captures both bull/bear trends AND range-bound periods
- 1d HTF filter prevents counter-trend trades that destroyed #685/#690
- Simple logic = more trades (avoid 0-trade failures like #699-#708)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_hma_rsi_bb_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

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
    return adx, atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - volatility bands around SMA."""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, sma, pct_b

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
    
    # Calculate primary (4h) indicators
    adx_4h, atr_4h = calculate_adx(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_sma, pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(adx_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION ===
        adx_strong = adx_4h[i] > 25  # Trending
        adx_weak = adx_4h[i] < 20    # Ranging
        regime_trend = adx_strong
        regime_range = adx_weak
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === TREND REGIME (ADX > 25) ===
        if regime_trend:
            # Long: Price above 1d HMA + RSI pullback + above SMA200
            if trend_bullish and rsi_4h[i] < 45 and above_sma200:
                desired_signal = current_size
            
            # Short: Price below 1d HMA + RSI bounce + below SMA200
            elif trend_bearish and rsi_4h[i] > 55 and below_sma200:
                desired_signal = -current_size
            
            # Weaker signals (single confirmation)
            elif trend_bullish and rsi_4h[i] < 35:
                desired_signal = REDUCED_SIZE
            elif trend_bearish and rsi_4h[i] > 65:
                desired_signal = -REDUCED_SIZE
        
        # === RANGE REGIME (ADX < 20) ===
        elif regime_range:
            # Long: RSI oversold + price near BB lower
            if rsi_4h[i] < 30 and pct_b[i] < 0.2:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI overbought + price near BB upper
            elif rsi_4h[i] > 70 and pct_b[i] > 0.8:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Only take strongest signals
            if rsi_4h[i] < 25 and trend_bullish:
                desired_signal = REDUCED_SIZE
            elif rsi_4h[i] > 75 and trend_bearish:
                desired_signal = -REDUCED_SIZE
        
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
                # Hold long if RSI not overbought and trend intact
                if rsi_4h[i] < 70 and trend_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if RSI not oversold and trend intact
                if rsi_4h[i] > 30 and trend_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if rsi_4h[i] > 75:
                desired_signal = 0.0
            elif close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi_4h[i] < 25:
                desired_signal = 0.0
            elif close[i] > hma_1d_aligned[i]:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
2026-03-23 13:04
