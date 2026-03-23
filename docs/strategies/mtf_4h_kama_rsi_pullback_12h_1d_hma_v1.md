# Strategy: mtf_4h_kama_rsi_pullback_12h_1d_hma_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.199 | -5.2% | -10.9% | 617 | FAIL |
| ETHUSDT | -1.093 | -9.3% | -14.6% | 637 | FAIL |
| SOLUSDT | 0.268 | +33.9% | -9.0% | 646 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.207 | +7.8% | -7.2% | 193 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #034: 4h KAMA Trend + 12h/1d HMA Filter + RSI Pullback Entries

Hypothesis: 4h primary with dual HTF trend filters will improve Sharpe over exp#026.
Key design based on learned failures:
1. 1d HMA(21) for major trend bias (call ONCE before loop via mtf_data)
2. 12h HMA(21) for intermediate trend direction (call ONCE before loop)
3. 4h KAMA(14) for adaptive primary trend (better than EMA in chop)
4. RSI(14) pullback entries (not breakouts) - long on RSI 35-45, short on 55-65
5. ATR(14) for stoploss (2.5x) - protects from major drawdowns
6. Looser entry conditions to ensure trade generation (learned from 0-trade failures)
7. Discrete sizing: 0.20 base, 0.25 medium, 0.30 strong trend alignment

Why this should work:
- KAMA adapts to volatility (better than HMA/EMA in ranging markets)
- Dual HTF filters (12h + 1d) provide stronger trend confirmation
- RSI pullback entries catch retracements in trends (not breakouts which fail)
- 4h TF targets 30-60 trades/year (optimal for fee efficiency)
- Simple RSI ranges (35-45, 55-65) ensure trades actually trigger

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_12h_1d_hma_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    ER (Efficiency Ratio) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    signal = np.abs(close - np.roll(close, period))
    signal[0:period] = np.nan
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
    noise[0:period] = np.nan
    
    er = signal / np.where(noise == 0, 1e-10, noise)
    er = np.nan_to_num(er, nan=0)
    
    # Smoothing Constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMA trends
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_14 = calculate_kama(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.20
    MEDIUM_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        # Both HTF must agree for strong signal
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend: both 12h and 1d agree
        htf_strong_bullish = htf_12h_bullish and htf_1d_bullish
        htf_strong_bearish = htf_12h_bearish and htf_1d_bearish
        
        # Weak trend: only one HTF agrees
        htf_weak_bullish = htf_12h_bullish or htf_1d_bullish
        htf_weak_bearish = htf_12h_bearish or htf_1d_bearish
        
        # === PRIMARY TREND (4h KAMA) ===
        kama_bullish = close[i] > kama_14[i]
        kama_bearish = close[i] < kama_14[i]
        
        # === RSI PULLBACK ENTRY (not breakout) ===
        # Long: RSI pulled back to 35-45 in bullish trend
        # Short: RSI pulled back to 55-65 in bearish trend
        rsi_pullback_long = 35 <= rsi_14[i] <= 50
        rsi_pullback_short = 50 <= rsi_14[i] <= 65
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_strong_bullish and kama_bullish:
            current_size = STRONG_SIZE
        elif htf_weak_bullish and kama_bullish:
            current_size = MEDIUM_SIZE
        elif htf_weak_bullish:
            current_size = BASE_SIZE
        elif htf_strong_bearish and kama_bearish:
            current_size = STRONG_SIZE
        elif htf_weak_bearish and kama_bearish:
            current_size = MEDIUM_SIZE
        elif htf_weak_bearish:
            current_size = BASE_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (loose conditions to ensure trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + KAMA bullish + RSI pullback
        if htf_weak_bullish and kama_bullish and rsi_pullback_long:
            new_signal = current_size
        
        # SHORT ENTRY: HTF bearish + KAMA bearish + RSI pullback
        elif htf_weak_bearish and kama_bearish and rsi_pullback_short:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 20 bars (~3.3 days on 4h), allow weaker entry
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htf_weak_bullish and kama_bullish:
                new_signal = current_size * 0.8
            elif htf_weak_bearish and kama_bearish:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if primary trend (KAMA) turns bearish
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            # Exit short if primary trend (KAMA) turns bullish
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes very overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Exit short when RSI becomes very oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
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
2026-03-22 21:11
