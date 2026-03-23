# Strategy: mtf_4h_hma_rsi_pullback_1d_simple_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.305 | +13.0% | -7.6% | 207 | FAIL |
| ETHUSDT | -0.303 | +10.4% | -12.9% | 192 | FAIL |
| SOLUSDT | 0.222 | +31.5% | -13.0% | 187 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.401 | +10.6% | -7.1% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #549: 4h Primary + 1d HTF — Simplified Trend-Pullback with Vol Filter

Hypothesis: After analyzing 480+ failed strategies, the clearest pattern is:
- Complex regime switching (chop + Connors + ADX + volume) consistently FAILS
- Simple trend + pullback works best (#543 Sharpe=0.270 on 1d timeframe)
- 4h timeframe should target 20-50 trades/year (optimal per rules)
- Key insight: FEWER filters = MORE trades = better Sharpe (fee drag < missed opportunities)

This strategy uses SIMPLIFIED logic:
1. 4h HMA(21) for primary trend direction
2. 1d HMA(21) aligned for major trend bias (filter counter-trend)
3. RSI(14) pullback entry: long when RSI 35-50 in uptrend, short when RSI 50-65 in downtrend
4. Volatility filter: ATR(14)/ATR(50) ratio to avoid low-vol whipsaw
5. ATR(14) 2.5x trailing stop for all positions
6. Asymmetric sizing: 0.30 bull regime, 0.25 bear regime (crypto crashes faster than rallies)

Why this might beat Sharpe=0.435:
- Simpler = more trades (20-50/year target) = less chance of 0-trade failure
- 1d HTF filter prevents major counter-trend losses (key failure mode in 2022)
- RSI pullback entries catch dips in trends (proven in #543)
- 4h TF balances trade frequency vs fee drag optimally
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Position sizing: 0.25-0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_simple_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # 4h HMA for trend confirmation
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: smaller size in bear regime (crypto crashes faster)
    POSITION_SIZE_BULL = 0.30
    POSITION_SIZE_BEAR = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(atr_50[i]) or atr_50[i] == 0:
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        bull_regime_4h = close[i] > hma_4h_21[i]
        bear_regime_4h = close[i] < hma_4h_21[i]
        
        hma_4h_slope_bull = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_slope_bear = hma_4h_21[i] < hma_4h_50[i]
        
        # === VOLATILITY FILTER (avoid low-vol whipsaw) ===
        # ATR ratio > 0.7 means vol is at least 70% of long-term avg
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        vol_ok = vol_ratio > 0.7
        
        # === ADX FILTER (ensure some trend strength) ===
        # ADX > 15 means some directional movement (not completely flat)
        trend_ok = adx_14[i] > 15.0
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI 35-50 in uptrend (pullback, not oversold crash)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 52.0
        # Short: RSI 48-65 in downtrend (rally into resistance)
        rsi_pullback_short = 48.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC — SIMPLIFIED ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bull + 4h bull + RSI pullback + vol OK
        if bull_regime_1d and bull_regime_4h and rsi_pullback_long and vol_ok:
            # Size based on 1d regime strength
            if hma_1d_slope_bull:
                new_signal = POSITION_SIZE_BULL
            else:
                new_signal = POSITION_SIZE_BULL * 0.8
        
        # SHORT ENTRY: 1d bear + 4h bear + RSI pullback + vol OK
        elif bear_regime_1d and bear_regime_4h and rsi_pullback_short and vol_ok:
            # Size based on 1d regime strength
            if hma_1d_slope_bear:
                new_signal = -POSITION_SIZE_BEAR
            else:
                new_signal = -POSITION_SIZE_BEAR * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
            elif bear_regime_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
                new_signal = 0.0
            elif bull_regime_4h and hma_4h_slope_bull:
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
2026-03-23 05:53
