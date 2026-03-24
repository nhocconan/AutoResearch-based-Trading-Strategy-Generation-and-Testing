# Strategy: mtf_12h_hma_rsi_1d_trend_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.125 | +9.6% | -18.5% | 603 | FAIL |
| ETHUSDT | -0.103 | +7.1% | -28.2% | 585 | FAIL |
| SOLUSDT | 0.642 | +117.8% | -35.6% | 575 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.198 | +8.8% | -14.3% | 176 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1506: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 1100+ failed strategies, the pattern is clear:
1. 12h timeframe should generate 20-50 trades/year (not 0 like complex filter strategies)
2. Complex filters (CRSI+CHOP+session+volume) = 0 trades (#1495, #1498, #1500)
3. SIMPLER works: 1d HMA for trend bias + 12h HMA/RSI for entry timing
4. 12h is the sweet spot: fewer trades than 1h (less fee drag), more signals than 1d
5. Proven pattern from #1505 (1h HMA+RSI) but adapted for 12h with 1d HTF

Key design choices:
- Use 1d HMA(21) for macro trend direction (HTF filter)
- Use 12h HMA(21) for primary trend confirmation
- Use 12h RSI(14) for pullback entries within trend
- Use ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (appropriate for 12h trade frequency)
- Discrete signal levels (0.0, ±0.30) to minimize fee churn
- LOOSE entry conditions to ensure trades happen (RSI 35-65 bands)

Timeframe: 12h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.30 (discrete: 0.0, ±0.30)
Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months), Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_1d_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 12h (fewer trades than 1h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) - confirmation ===
        h12_bull = close[i] > hma_12h[i]
        h12_bear = close[i] < hma_12h[i]
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === RSI PULLBACK - LOOSE bands for MORE trades ===
        # Long: RSI pulled back but not oversold (35-55)
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-65)
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 12h ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 12h bullish + RSI pullback
        # Option 1: Strong trend (1d + 12h both bull) + RSI pullback
        if daily_bull and h12_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Option 2: 1d bull + 12h bull + above SMA50 (looser, ensures trades)
        elif daily_bull and h12_bull and above_sma50:
            desired_signal = BASE_SIZE * 0.8
        # Option 3: 1d bull + 12h above HMA + RSI not overbought (loosest)
        elif daily_bull and h12_bull and rsi[i] < 60.0:
            desired_signal = BASE_SIZE * 0.6
        # Option 4: 1d bull only + strong RSI support (fallback for trades)
        elif daily_bull and rsi[i] < 50.0 and above_sma50:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT: 1d bearish + 12h bearish + RSI pullback
        # Option 1: Strong trend (1d + 12h both bear) + RSI pullback
        elif daily_bear and h12_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Option 2: 1d bear + 12h bear + below SMA50 (looser, ensures trades)
        elif daily_bear and h12_bear and below_sma50:
            desired_signal = -BASE_SIZE * 0.8
        # Option 3: 1d bear + 12h below HMA + RSI not oversold (loosest)
        elif daily_bear and h12_bear and rsi[i] > 40.0:
            desired_signal = -BASE_SIZE * 0.6
        # Option 4: 1d bear only + strong RSI resistance (fallback for trades)
        elif daily_bear and rsi[i] > 50.0 and below_sma50:
            desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 01:05
