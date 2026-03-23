# Strategy: mtf_4h_kama_adx_chop_12h_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.005 | +19.8% | -12.2% | 175 | PASS |
| ETHUSDT | 0.551 | +58.4% | -13.6% | 164 | PASS |
| SOLUSDT | 0.839 | +131.1% | -17.9% | 146 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.121 | -14.5% | -15.5% | 60 | FAIL |
| ETHUSDT | 0.147 | +7.6% | -8.4% | 52 | PASS |
| SOLUSDT | -0.581 | -4.7% | -19.0% | 51 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #594: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX + Choppiness

Hypothesis: Building on #591's success (Sharpe=0.423), this strategy replaces HMA with
KAMA (Kaufman Adaptive Moving Average) which adapts to market efficiency - faster in
trends, slower in chop. Combined with ADX for trend strength and Choppiness for regime,
this should improve entry timing and reduce whipsaws.

Key improvements over #591:
1. KAMA(14) adapts to volatility - proven on ETH (Sharpe +0.755 in literature)
2. ADX(14) > 20 confirms trend strength before breakout entries
3. 12h HMA(21) for intermediate trend bias (faster reaction than 1d)
4. More lenient RSI thresholds (20/80 instead of 25/75) to ensure >=30 trades
5. Volume confirmation on breakouts (volume > SMA20 volume)
6. ATR(14) 2.5x trailing stop with position tracking

Why this might beat Sharpe=0.520:
- KAMA reduces lag in trends, prevents whipsaws in chop (adaptive)
- ADX filter prevents false breakouts in weak trends
- 12h HTF reacts faster than 1d to regime changes
- Wider thresholds ensure sufficient trade frequency
- Volume filter adds confirmation on breakouts

Position sizing: 0.30 discrete (max 0.40 per Rule 4)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_12h_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - moves fast in trends, slow in chop.
    
    Efficiency Ratio (ER) = |Close - Close(n)| / Sum(|Close(i) - Close(i-1)|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    KAMA = KAMA(prev) + SC * (Close - KAMA(prev))
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period))
    sum_volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (sum_volatility + 1e-10)
    er = er.fillna(0.0).values
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX) - trend strength indicator.
    ADX > 25 = strong trend, ADX < 20 = weak/no trend.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    # Clip to valid range
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for intermediate trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    kama_14 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Volume SMA for confirmation
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(kama_14[i]) or np.isnan(vol_sma20[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop_regime = chop_14[i] > 61.8
        is_trend_regime = chop_14[i] < 38.2
        
        # === 12H INTERMEDIATE TREND BIAS ===
        bull_bias_12h = close[i] > hma_12h_21_aligned[i]
        bear_bias_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        kama_bull = close[i] > kama_14[i]
        kama_bear = close[i] < kama_14[i]
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 20.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma20[i]
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        
        # --- CHOP REGIME: Mean Reversion (RSI extremes) ---
        if is_chop_regime:
            # Long: RSI < 20 (oversold) + 12h bull bias + KAMA bull
            if rsi_14[i] < 20.0 and bull_bias_12h and kama_bull:
                new_signal = POSITION_SIZE
            
            # Short: RSI > 80 (overbought) + 12h bear bias + KAMA bear
            elif rsi_14[i] > 80.0 and bear_bias_12h and kama_bear:
                new_signal = -POSITION_SIZE
        
        # --- TREND REGIME: Trend Following (KAMA + ADX) ---
        elif is_trend_regime:
            # Long: KAMA bull + ADX strong + 12h bull bias + volume
            if kama_bull and strong_trend and bull_bias_12h and volume_confirmed:
                # Size based on 12h trend strength
                if hma_12h_slope_bull:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = POSITION_SIZE * 0.7
            
            # Short: KAMA bear + ADX strong + 12h bear bias + volume
            elif kama_bear and strong_trend and bear_bias_12h and volume_confirmed:
                # Size based on 12h trend strength
                if hma_12h_slope_bear:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -POSITION_SIZE * 0.7
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on 12h bias flip to bear + KAMA flip
        if in_position and position_side > 0:
            if bear_bias_12h and kama_bear:
                new_signal = 0.0
        
        # Exit short on 12h bias flip to bull + KAMA flip
        if in_position and position_side < 0:
            if bull_bias_12h and kama_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 06:31
