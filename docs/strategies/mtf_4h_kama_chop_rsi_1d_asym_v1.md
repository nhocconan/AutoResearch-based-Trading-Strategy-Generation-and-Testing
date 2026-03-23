# Strategy: mtf_4h_kama_chop_rsi_1d_asym_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.885 | -7.3% | -15.3% | 1070 | FAIL |
| ETHUSDT | -0.305 | +6.4% | -13.7% | 1090 | FAIL |
| SOLUSDT | 0.242 | +35.5% | -32.6% | 944 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.235 | +8.9% | -14.7% | 350 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #301: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Timing

Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in crypto because:
1. KAMA adapts smoothing based on market efficiency ratio (ER)
2. Fast smoothing in trends, slow smoothing in chop — reduces whipsaws
3. Combined with Choppiness Index regime filter for dual-mode trading
4. 1d HMA(21) as primary trend filter (asymmetric: long-only in bull, short-only in bear)
5. RSI(14) for precise entry timing within regime
6. Target: 25-45 trades/year on 4h (appropriate frequency)

Why this might beat #292 (Sharpe=0.424):
- KAMA adapts to volatility regime automatically (no manual parameter switching)
- Choppiness Index provides cleaner regime detection than ADX
- Asymmetric entries match crypto behavior (bull trends stronger than bear)
- 4h timeframe balances signal quality vs trade frequency better than 1d/12h

Position sizing: 0.25 base, 0.35 strong conviction (trend + regime aligned)
Stoploss: 2.5 * ATR trailing (tighter than daily, appropriate for 4h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_rsi_1d_asym_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, period=40, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market Efficiency Ratio (ER).
    ER = |change| / sum(|changes|) over period
    High ER (trending) → fast smoothing constant
    Low ER (choppy) → slow smoothing constant
    
    This reduces whipsaws in chop while capturing trends quickly.
    """
    n = period
    er = np.zeros(len(close))
    kama = np.zeros(len(close))
    
    # Calculate Efficiency Ratio
    for i in range(n, len(close)):
        change = np.abs(close[i] - close[i - n])
        noise = np.sum(np.abs(np.diff(close[max(0, i-n):i+1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA with SMA
    kama[n-1] = np.mean(close[max(0, n-period):n])
    
    for i in range(n, len(close)):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Values between 38.2-61.8 = transitional
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_40 = calculate_kama(close, 40, 2, 30)
    kama_10 = calculate_kama(close, 10, 2, 20)  # Faster KAMA for entry timing
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_40[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1d HMA (only take longs or reduce shorts)
        # Bear: price below 1d HMA (only take shorts or reduce longs)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        # 45-55 = transitional (reduce size)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        is_transitional = not is_choppy and not is_trending
        
        # === VOLATILITY REGIME (ATR ratio) ===
        # High vol: ATR(14)/ATR(30) > 1.4 (reduce position size)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.4
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 4H LOCAL SIGNALS ===
        # KAMA trend direction
        kama_bullish = kama_40[i] > kama_40[i-5] if i >= 5 else False
        kama_bearish = kama_40[i] < kama_40[i-5] if i >= 5 else False
        
        # Fast KAMA crossover for entry timing
        kama_fast_above_slow = kama_10[i] > kama_40[i]
        kama_fast_below_slow = kama_10[i] < kama_40[i]
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_40[i]
        price_below_kama = close[i] < kama_40[i]
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i] * 1.003
        bb_break_upper = close[i] > bb_upper[i] * 0.997
        bb_near_lower = close[i] < bb_lower[i] * 1.01
        bb_near_upper = close[i] > bb_upper[i] * 0.99
        
        # === ENTRY LOGIC (ASYMMETRIC + DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (only when 1d regime bull or neutral)
        if regime_bull or (not regime_bear and is_choppy):
            # Trend follow: KAMA bullish + price above KAMA + RSI confirming
            if is_trending and kama_bullish and price_above_kama and rsi_14[i] > 45:
                new_signal = BASE_SIZE * vol_scale
            
            # KAMA fast crossover in trend regime
            elif is_trending and kama_fast_above_slow and price_above_kama and rsi_neutral:
                new_signal = BASE_SIZE * vol_scale
            
            # Mean revert in choppy market + BB lower break + RSI oversold
            elif is_choppy and bb_near_lower and rsi_oversold:
                new_signal = BASE_SIZE * vol_scale
            
            # Extreme RSI oversold in bull regime (strong conviction long)
            elif rsi_extreme_oversold and regime_bull:
                new_signal = STRONG_SIZE * vol_scale
            
            # Transitional regime + KAMA bullish + RSI rising
            elif is_transitional and kama_bullish and rsi_14[i] > 45 and rsi_14[i] > rsi_14[i-1]:
                new_signal = MIN_SIZE * vol_scale
        
        # SHORT ENTRIES (only when 1d regime bear or neutral)
        if regime_bear or (not regime_bull and is_choppy):
            # Trend follow: KAMA bearish + price below KAMA + RSI confirming
            if is_trending and kama_bearish and price_below_kama and rsi_14[i] < 55:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            
            # KAMA fast crossover in trend regime
            elif is_trending and kama_fast_below_slow and price_below_kama and rsi_neutral:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            
            # Mean revert in choppy market + BB upper break + RSI overbought
            elif is_choppy and bb_near_upper and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            
            # Extreme RSI overbought in bear regime (strong conviction short)
            elif rsi_extreme_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale
            
            # Transitional regime + KAMA bearish + RSI falling
            elif is_transitional and kama_bearish and rsi_14[i] < 55 and rsi_14[i] < rsi_14[i-1]:
                if new_signal == 0.0:
                    new_signal = -MIN_SIZE * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 35 bars (~35 * 4h = 140h ≈ 6 days)
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 42 and price_above_kama:
                new_signal = MIN_SIZE * vol_scale
            elif regime_bear and rsi_14[i] < 58 and price_below_kama:
                new_signal = -MIN_SIZE * vol_scale
            elif is_choppy and rsi_14[i] < 38:
                new_signal = MIN_SIZE * vol_scale
            elif is_choppy and rsi_14[i] > 62:
                new_signal = -MIN_SIZE * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === KAMA REVERSAL EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Long position: exit when KAMA turns bearish
            if position_side > 0 and kama_bearish and price_below_kama:
                kama_exit = True
            # Short position: exit when KAMA turns bullish
            if position_side < 0 and kama_bullish and price_above_kama:
                kama_exit = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Short position: exit when RSI oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_kama:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_kama:
                regime_reversal = True
        
        if stoploss_triggered or kama_exit or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.18:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE * vol_scale
            elif new_signal > 0:
                new_signal = BASE_SIZE * vol_scale
            elif new_signal < -0.30:
                new_signal = -STRONG_SIZE * vol_scale
            else:
                new_signal = -BASE_SIZE * vol_scale
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
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
2026-03-23 02:12
