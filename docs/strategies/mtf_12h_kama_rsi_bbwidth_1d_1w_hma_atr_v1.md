# Strategy: mtf_12h_kama_rsi_bbwidth_1d_1w_hma_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.300 | -23.4% | -29.6% | 379 | FAIL |
| ETHUSDT | -0.879 | -21.6% | -35.0% | 412 | FAIL |
| SOLUSDT | 0.440 | +56.2% | -20.9% | 425 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.660 | +19.0% | -15.5% | 147 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #245: 12h Regime-Adaptive Strategy with KAMA + RSI + BB Width + 1d/1w HMA

Hypothesis: 12h timeframe captures multi-day swings while filtering noise. 
Using KAMA (Kaufman Adaptive MA) for trend detection + RSI for entry timing +
Bollinger Band Width for regime detection + 1d/1w HMA for higher timeframe bias.

Why this might work on 12h:
- 12h balances signal quality (less noise than 1h/4h) with trade frequency (more than 1d)
- KAMA adapts to volatility - smooth in ranges, responsive in trends
- BB Width percentile detects squeeze/breakout regimes
- 1d HMA provides trend bias, 1w HMA provides macro bias
- RSI(7) extremes with regime filter = high-probability entries
- Conservative sizing (0.30) + ATR stoploss controls drawdown

Key improvements over failed experiments:
- #239 (12h vol spike): Sharpe=-1.138 - too strict on vol spikes
- #233 (12h KAMA): Sharpe=-0.029 - missing regime filter
- This uses looser RSI thresholds (25/75 vs 20/80) for more trades
- BB Width regime filter prevents entries in wrong regime
- Dual HTF (1d + 1w) for stronger trend confirmation

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_bbwidth_1d_1w_hma_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period]
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100  # Bandwidth as percentage
    
    return upper.values, lower.values, sma.values, bb_width.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    
    return upper, lower

def calculate_zscore(series, period=20):
    """Calculate Z-score for mean reversion detection."""
    series_s = pd.Series(series)
    rolling_mean = series_s.rolling(window=period, min_periods=period).mean()
    rolling_std = series_s.rolling(window=period, min_periods=period).std()
    zscore = (series_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_30 = calculate_kama(close, er_period=10, fast_period=2, slow_period=60)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate BB Width percentile for regime detection
    bb_width_percentile = np.zeros(n)
    bb_width_s = pd.Series(bb_width)
    for i in range(50, n):
        window = bb_width_s.iloc[i-50:i]
        if len(window.dropna()) > 0:
            bb_width_percentile[i] = 100 * (window < bb_width[i]).sum() / len(window.dropna())
    
    # Calculate price Z-score for mean reversion
    price_zscore = calculate_zscore(close, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_price_idx = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = intermediate trend bias
        # 1w HMA = macro trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        strong_bull = bull_trend_1d and bull_trend_1w
        strong_bear = bear_trend_1d and bear_trend_1w
        
        # === REGIME DETECTION ===
        # BB Width percentile < 20 = Squeeze (expecting breakout)
        # BB Width percentile > 80 = Extended (expecting mean reversion)
        # 20-80 = Normal
        is_squeeze = bb_width_percentile[i] < 25
        is_extended = bb_width_percentile[i] > 75
        
        # KAMA trend state
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # KAMA slope (5-bar lookback)
        kama_slope_bullish = kama_10[i] > kama_10[i-5] if i >= 5 else False
        kama_slope_bearish = kama_10[i] < kama_10[i-5] if i >= 5 else False
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- TREND FOLLOWING ENTRIES (KAMA crossover + HTF bias) ---
        # Long: KAMA bullish + 1d bullish + RSI not overbought
        if kama_bullish and kama_slope_bullish and bull_trend_1d:
            if rsi_7[i] < 70:  # Not overbought
                new_signal = SIZE_BASE
        
        # Short: KAMA bearish + 1d bearish + RSI not oversold
        if kama_bearish and kama_slope_bearish and bear_trend_1d:
            if rsi_7[i] > 30:  # Not oversold
                new_signal = -SIZE_BASE
        
        # --- MEAN REVERSION ENTRIES (BB extended + RSI extreme) ---
        # Only when HTF bias is neutral or supportive
        if is_extended:
            # Long: Price at BB lower + RSI oversold + 1d not strongly bearish
            if close[i] < bb_lower[i] and rsi_7[i] < 30:
                if not strong_bear:
                    new_signal = SIZE_BASE
            
            # Short: Price at BB upper + RSI overbought + 1d not strongly bullish
            if close[i] > bb_upper[i] and rsi_7[i] > 70:
                if not strong_bull:
                    new_signal = -SIZE_BASE
        
        # --- BREAKOUT ENTRIES (Donchian + squeeze) ---
        if is_squeeze:
            # Long breakout: Price breaks Donchian upper + KAMA turning bullish
            if close[i] > donchian_upper[i] and kama_bullish:
                new_signal = SIZE_BASE
            
            # Short breakout: Price breaks Donchian lower + KAMA turning bearish
            if close[i] < donchian_lower[i] and kama_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
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
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
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
2026-03-22 14:22
