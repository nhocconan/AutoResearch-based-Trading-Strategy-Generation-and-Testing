# Strategy: mtf_1d_chop_regime_1w_hma_rsi_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.006 | +20.8% | -7.5% | 226 | PASS |
| ETHUSDT | -0.457 | +4.7% | -8.9% | 231 | FAIL |
| SOLUSDT | 0.305 | +39.2% | -28.1% | 222 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.955 | -11.5% | -16.4% | 64 | FAIL |
| SOLUSDT | 0.056 | +6.4% | -6.3% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #037: 1d Choppiness Regime + 1w HMA Trend + RSI Entries

Hypothesis: Daily timeframe with weekly trend filter and regime-adaptive entries
will reduce whipsaws while maintaining trade frequency.

Key design:
1. 1w HMA(21) for major trend bias (call ONCE via mtf_data)
2. Choppiness Index(14) for regime detection (>55 = range, <45 = trend)
3. RSI(14) for entry timing with regime-specific thresholds (wide ranges)
4. ATR(14) for stoploss (2.5x)
5. Discrete sizing: 0.25 base, 0.30 strong trend

Why this should work:
- 1d TF naturally limits trades to 20-50/year (fee efficient)
- 1w HTF filter prevents counter-trend trades in strong trends
- Choppiness adapts between mean-revert and trend-follow modes
- RSI thresholds wide enough (35-55 long, 45-65 short) to ensure trades trigger
- Frequency safeguard after 30 bars without trades

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_1w_hma_rsi_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
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
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        # CHOP > 55 = ranging (mean revert)
        # CHOP < 45 = trending (trend follow)
        # 45 - 55 = neutral (use trend bias)
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === ENTRY LOGIC - REGIME ADAPTIVE (wide thresholds for trade gen) ===
        new_signal = 0.0
        
        if is_trending and htf_bullish:
            # Trend follow long: RSI pullback in uptrend (wide range)
            if 30 <= rsi_14[i] <= 60:
                new_signal = STRONG_SIZE
        
        elif is_trending and htf_bearish:
            # Trend follow short: RSI rally in downtrend (wide range)
            if 40 <= rsi_14[i] <= 70:
                new_signal = -STRONG_SIZE
        
        elif is_choppy:
            # Mean reversion in range (wider thresholds)
            if rsi_14[i] < 40:
                new_signal = BASE_SIZE  # long at oversold
            elif rsi_14[i] > 60:
                new_signal = -BASE_SIZE  # short at overbought
        
        else:
            # Neutral regime: use HTF bias with moderate RSI
            if htf_bullish and rsi_14[i] < 55:
                new_signal = BASE_SIZE
            elif htf_bearish and rsi_14[i] > 45:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~25 days on 1d), force entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if htf_bullish:
                new_signal = BASE_SIZE * 0.8
            elif htf_bearish:
                new_signal = -BASE_SIZE * 0.8
        
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
            # Exit long if HTF trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if HTF trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes very overbought
            if position_side > 0 and rsi_14[i] > 80:
                rsi_exit = True
            # Exit short when RSI becomes very oversold
            if position_side < 0 and rsi_14[i] < 20:
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
2026-03-22 21:14
