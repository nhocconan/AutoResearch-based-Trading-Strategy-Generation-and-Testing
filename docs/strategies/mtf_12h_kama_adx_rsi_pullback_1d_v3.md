# Strategy: mtf_12h_kama_adx_rsi_pullback_1d_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.313 | -1.1% | -21.8% | 300 | FAIL |
| ETHUSDT | -0.083 | +8.1% | -19.3% | 305 | FAIL |
| SOLUSDT | 0.874 | +167.6% | -22.0% | 330 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.124 | +6.9% | -15.2% | 98 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #536: 12h Primary + 1d HTF — KAMA Trend + ADX Filter + RSI Pullback

Hypothesis: After 480+ failed strategies, return to proven components with 
simpler logic that ensures trade frequency across all symbols.

Key insights from failures:
- Complex volspike/Fisher/Choppiness combos = 0 trades or negative Sharpe
- Simpler trend-following with pullback entries works better
- 12h TF should target 20-50 trades/year (optimal for fee/trade ratio)
- KAMA adapts to market conditions better than fixed EMA/HMA
- ADX filter avoids choppy market whipsaws

This strategy uses:
1. 1d HMA(21) for major trend direction (HTF filter)
2. 12h KAMA(14) for adaptive trend following (faster in trends, slower in ranges)
3. ADX(14) > 20 to confirm trending market (avoid chop)
4. RSI(14) pullback entries (30-40 for long, 60-70 for short)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete position sizing (0.28) to minimize fee churn

Why this might work:
- KAMA's adaptive nature handles both bull/bear/range markets
- 1d trend filter prevents counter-trend trades (major failure mode)
- ADX filter avoids whipsaws in choppy conditions
- RSI pullback entries avoid chasing breakouts
- 12h TF balances trade frequency with signal quality
- Simple logic = consistent signals across BTC/ETH/SOL

Position sizing: 0.28 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_rsi_pullback_1d_v3"
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

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - faster in trends, slower in ranges.
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    fast_sc = 2/(fast+1), slow_sc = 2/(slow+1)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = 0
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = np.zeros(n)
    er[period:] = change[period:] / (volatility[period:] + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate +DM and -DM
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # When both are positive, keep the larger one
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth +DM, -DM, and TR
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # DX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_14 = calculate_kama(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track KAMA slope
    prev_kama = np.zeros(n)
    prev_kama[1:] = kama_14[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(kama_14[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength confirmation
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        # KAMA slope (adaptive trend)
        kama_slope_bull = kama_14[i] > prev_kama[i]
        kama_slope_bear = kama_14[i] < prev_kama[i]
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_14[i]
        price_below_kama = close[i] < kama_14[i]
        
        # === ADX FILTER (trending market only) ===
        trending = adx_14[i] > 20.0  # Market is trending
        strong_trend = adx_14[i] > 25.0  # Strong trend
        
        # === RSI PULLBACK FILTER ===
        rsi_pullback_long = 30.0 < rsi_14[i] < 50.0  # Pullback in uptrend
        rsi_pullback_short = 50.0 < rsi_14[i] < 70.0  # Pullback in downtrend
        rsi_oversold = rsi_14[i] < 35.0  # Good for long entry
        rsi_overbought = rsi_14[i] > 65.0  # Good for short entry
        
        # === ENTRY LOGIC — MULTIPLE CONDITIONS FOR FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (OR logic for frequency)
        # Condition 1: Bull regime + trending + KAMA bull + RSI pullback
        if bull_regime and trending and kama_slope_bull and rsi_pullback_long:
            new_signal = POSITION_SIZE
        # Condition 2: Bull regime + strong trend + price above KAMA + RSI not overbought
        elif bull_regime and strong_trend and price_above_kama and rsi_14[i] < 65.0:
            new_signal = POSITION_SIZE
        # Condition 3: Bull regime + KAMA bull + RSI oversold (deep pullback)
        elif bull_regime and kama_slope_bull and rsi_oversold:
            new_signal = POSITION_SIZE
        # Condition 4: Strong bull (1d HMA slope) + trending + KAMA bull
        elif bull_regime and hma_slope_bull and trending and kama_slope_bull:
            new_signal = POSITION_SIZE * 0.8
        # Condition 5: Bull regime + price above KAMA + ADX rising (momentum)
        elif bull_regime and price_above_kama and adx_14[i] > adx_14[i-1] if i > 0 else False:
            new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + trending + KAMA bear + RSI pullback
            if bear_regime and trending and kama_slope_bear and rsi_pullback_short:
                new_signal = -POSITION_SIZE
            # Condition 2: Bear regime + strong trend + price below KAMA + RSI not oversold
            elif bear_regime and strong_trend and price_below_kama and rsi_14[i] > 35.0:
                new_signal = -POSITION_SIZE
            # Condition 3: Bear regime + KAMA bear + RSI overbought (deep bounce)
            elif bear_regime and kama_slope_bear and rsi_overbought:
                new_signal = -POSITION_SIZE
            # Condition 4: Strong bear (1d HMA slope) + trending + KAMA bear
            elif bear_regime and hma_slope_bear and trending and kama_slope_bear:
                new_signal = -POSITION_SIZE * 0.8
            # Condition 5: Bear regime + price below KAMA + ADX rising (momentum)
            elif bear_regime and price_below_kama and adx_14[i] > adx_14[i-1] if i > 0 else False:
                new_signal = -POSITION_SIZE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip or weak trend) ===
        # Exit long on regime flip to bear or ADX drops (trend weakening)
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif adx_14[i] < 15.0:  # Trend weakening
                new_signal = 0.0
        
        # Exit short on regime flip to bull or ADX drops
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif adx_14[i] < 15.0:  # Trend weakening
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
2026-03-23 05:41
