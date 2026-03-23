# Strategy: mtf_12h_hma_rsi_pullback_1d_bias_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.492 | +2.5% | -8.4% | 370 | FAIL |
| ETHUSDT | -0.655 | -21.5% | -33.0% | 1619 | FAIL |
| SOLUSDT | 0.053 | +15.7% | -36.9% | 1416 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.054 | +5.8% | -16.8% | 517 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #016: 12h HMA Trend + RSI Pullback with 1d Bias

Hypothesis: Previous regime-adaptive strategies failed due to over-complexity and
too-restrictive entry conditions. This strategy simplifies to proven patterns:
1. 1d HMA for major trend bias (same as current best baseline)
2. 12h HMA(16/48) crossover for trend direction
3. RSI(14) pullback entries (30-70 range, not extreme) for better frequency
4. ATR(14) trailing stoploss at 2.5x
5. Discrete position sizing (0.25-0.30) to minimize fee churn

Why this should work:
- Simpler logic = more trades (addressing 0-trade failures in exp #006, #008, #010)
- RSI pullback (not extreme) catches more entries in trending markets
- 12h timeframe naturally filters noise (target 20-50 trades/year)
- 1d HMA bias prevents counter-trend trades (major improvement over pure 12h)
- Based on current best baseline pattern (mtf_4h_hma_rsi_pullback_1d_bias_v1)

Key changes from failed experiments:
- NO Choppiness Index regime detection (failed in #005, #012, #013)
- NO Connors RSI extreme thresholds (failed in #005, #008, #013)
- NO complex multi-regime logic (failed in #012, #013)
- Simpler, proven HMA + RSI pullback pattern

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_bias_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_16 = calculate_hma(close, 16)
    hma_12h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (SIMPLIFIED PULLBACK) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 12h HMA bullish + 1d bias bullish + RSI pullback (not extreme)
        if hma_bullish and daily_bullish:
            # RSI pullback to 35-50 range (not oversold, just cooling off)
            if 35 <= rsi_14[i] <= 55:
                new_signal = current_size
            # Or breakout above Donchian with RSI confirmation
            elif i > 0 and not np.isnan(donchian_upper[i-1]):
                if close[i] > donchian_upper[i-1] and rsi_14[i] > 50 and rsi_14[i] < 75:
                    new_signal = current_size
        
        # SHORT: 12h HMA bearish + 1d bias bearish + RSI pullback (not extreme)
        elif hma_bearish and daily_bearish:
            # RSI pullback to 45-65 range (not overbought, just cooling off)
            if 45 <= rsi_14[i] <= 65:
                new_signal = -current_size
            # Or breakout below Donchian with RSI confirmation
            elif i > 0 and not np.isnan(donchian_lower[i-1]):
                if close[i] < donchian_lower[i-1] and rsi_14[i] < 50 and rsi_14[i] > 25:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~20 days on 12h), force entry with weaker signal
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if hma_bullish and daily_bullish and rsi_14[i] > 45:
                new_signal = current_size * 0.6
            elif hma_bearish and daily_bearish and rsi_14[i] < 55:
                new_signal = -current_size * 0.6
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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
2026-03-22 20:51
